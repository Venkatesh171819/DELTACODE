#!/usr/bin/env python3
"""
allocate.py - Maximum bipartite matching with role capacities (workforce allocation).

Engine (hand-written, no library matching code):
  * Flow network G' = (V', E'):  s -> u (cap 1),  u -> v (cap 1),  v -> t (cap c_v)
  * Hopcroft-Karp / Dinic: BFS level graph + DFS blocking flow (iterative, current-arc pointers)
  * Warm start from a greedy assignment, so every augmenting path of length > 3 is
    exactly a "multi-step reassignment chain" that undoes a greedy misallocation.
  * Optimality certificate (Berge): residual BFS from s cannot reach t (no augmenting path)

scipy.optimize (milp / linprog) is used only to benchmark; networkx only for validation + drawing.

Usage:
  python allocate.py                       # synthetic seeded data (2400 candidates, 320 roles)
  python allocate.py --edges edges.csv --roles roles.csv   # real data
      edges.csv: candidate_id,role_id   (row order per candidate = preference order)
      roles.csv: role_id,capacity
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from collections import deque, defaultdict

import numpy as np


# ----------------------------------------------------------------------------- data
def generate(seed=2026, n_cand=2400, n_roles=320):
    rng = random.Random(seed)
    caps = [rng.choice([3, 4, 5, 6, 7, 8, 10]) for _ in range(n_roles)]
    # skewed popularity: a few "hot" roles are eligible for many candidates
    w = [1.0 / (i + 1) ** 0.7 for i in range(n_roles)]
    rng.shuffle(w)
    cand_adj = []
    for _ in range(n_cand):
        k = rng.choices([1, 2, 3, 4, 5], weights=[22, 30, 25, 15, 8])[0]
        chosen = []
        while len(chosen) < k:
            r = rng.choices(range(n_roles), weights=w)[0]
            if r not in chosen:
                chosen.append(r)
        cand_adj.append(chosen)          # order = candidate's preference order
    return cand_adj, caps


def _check_columns(reader, needed, path):
    missing = [c for c in needed if c not in (reader.fieldnames or [])]
    if missing:
        raise ValueError(f"{path}: missing column(s) {missing}; found {reader.fieldnames}")


def load_csv(edges_path, roles_path, candidates_path=None):
    """Read real data. Returns (cand_adj, caps, cand_ids, role_ids) with the ORIGINAL ids kept.
    Validation (raises ValueError with a readable message):
      * roles.csv needs role_id,capacity; capacities must be integers >= 0; role ids unique
      * edges.csv needs candidate_id,role_id; every role id must exist in roles.csv
    Tolerated: duplicate edges (collapsed, first-seen preference order kept), zero-capacity roles,
    blank ids are rejected. Candidates with no edge are kept (as unassignable) if you pass
    candidates_path (a csv with a candidate_id column)."""
    roles = {}
    with open(roles_path, newline="") as f:
        rd = csv.DictReader(f)
        _check_columns(rd, ["role_id", "capacity"], roles_path)
        for line, row in enumerate(rd, start=2):
            rid_ = (row["role_id"] or "").strip()
            if not rid_:
                raise ValueError(f"{roles_path}:{line}: empty role_id")
            if rid_ in roles:
                raise ValueError(f"{roles_path}:{line}: duplicate role_id {rid_!r}")
            try:
                cap = int(str(row["capacity"]).strip())
            except ValueError:
                raise ValueError(f"{roles_path}:{line}: capacity {row['capacity']!r} is not an integer")
            if cap < 0:
                raise ValueError(f"{roles_path}:{line}: negative capacity for {rid_!r}")
            roles[rid_] = cap
    role_ids = list(roles)
    rix = {r: i for i, r in enumerate(role_ids)}
    caps = [roles[r] for r in role_ids]

    cand_ids, adj = [], {}
    if candidates_path:
        with open(candidates_path, newline="") as f:
            rd = csv.DictReader(f)
            _check_columns(rd, ["candidate_id"], candidates_path)
            for line, row in enumerate(rd, start=2):
                c = (row["candidate_id"] or "").strip()
                if not c:
                    raise ValueError(f"{candidates_path}:{line}: empty candidate_id")
                if c not in adj:
                    cand_ids.append(c); adj[c] = []
    with open(edges_path, newline="") as f:
        rd = csv.DictReader(f)
        _check_columns(rd, ["candidate_id", "role_id"], edges_path)
        for line, row in enumerate(rd, start=2):
            c = (row["candidate_id"] or "").strip()
            r = (row["role_id"] or "").strip()
            if not c or not r:
                raise ValueError(f"{edges_path}:{line}: empty candidate_id or role_id")
            if r not in rix:
                raise ValueError(f"{edges_path}:{line}: role_id {r!r} is not listed in {roles_path}")
            if c not in adj:
                cand_ids.append(c); adj[c] = []
            if rix[r] not in adj[c]:                       # collapse duplicate edges
                adj[c].append(rix[r])
    return [adj[c] for c in cand_ids], caps, cand_ids, role_ids


# ----------------------------------------------------------------------------- greedy baseline
def greedy(cand_adj, caps):
    load = [0] * len(caps)
    match = [-1] * len(cand_adj)
    for u, nb in enumerate(cand_adj):          # irreversible first-fit
        for v in nb:
            if load[v] < caps[v]:
                match[u] = v
                load[v] += 1
                break
    return match


# ----------------------------------------------------------------------------- Hopcroft-Karp (Dinic form)
class Network:
    """s = 0, candidates 1..U, roles U+1..U+V, t = U+V+1. Edge e and e^1 are mutual residuals."""

    def __init__(self, cand_adj, caps):
        self.U, self.V = len(cand_adj), len(caps)
        self.n = self.U + self.V + 2
        self.s, self.t = 0, self.n - 1
        self.to, self.cap, self.adj = [], [], [[] for _ in range(self.n)]
        self.cv_edge = {}
        for u in range(self.U):
            self._add(self.s, 1 + u, 1)
        self.uv_edge = {}
        for u, nb in enumerate(cand_adj):
            for v in nb:
                self.uv_edge[(u, v)] = self._add(1 + u, 1 + self.U + v, 1)
        for v, c in enumerate(caps):
            self._add(1 + self.U + v, self.t, c)
        self.orig_cap = list(self.cap)

    def _add(self, a, b, c):
        e = len(self.to)
        self.to += [b, a]
        self.cap += [c, 0]
        self.adj[a].append(e)
        self.adj[b].append(e + 1)
        return e

    def push_initial(self, cand_adj, match):
        """Warm start: saturate the s-u-v-t path of every already-assigned candidate."""
        for u, v in enumerate(match):
            if v < 0:
                continue
            e = self.uv_edge[(u, v)]
            self.cap[e] -= 1; self.cap[e ^ 1] += 1
            es = self.adj[self.s][u]                      # s->u edge is the u-th added
            self.cap[es] -= 1; self.cap[es ^ 1] += 1
            et = self.adj[1 + self.U + v][-1]             # v->t edge is the last in v's list
            self.cap[et] -= 1; self.cap[et ^ 1] += 1

    def bfs(self):
        level = [-1] * self.n
        level[self.s] = 0
        q = deque([self.s])
        while q:
            x = q.popleft()
            for e in self.adj[x]:
                y = self.to[e]
                if self.cap[e] > 0 and level[y] < 0:
                    level[y] = level[x] + 1
                    q.append(y)
        return level

    def blocking_flow(self, level, trace):
        it = [0] * self.n
        flow = 0
        s, t = self.s, self.t
        while True:
            path, x = [], s
            while x != t:
                moved = False
                while it[x] < len(self.adj[x]):
                    e = self.adj[x][it[x]]
                    y = self.to[e]
                    if self.cap[e] > 0 and level[y] == level[x] + 1:
                        path.append(e); x = y; moved = True
                        break
                    it[x] += 1
                if not moved:
                    if x == s:
                        return flow
                    level[x] = -1                  # dead end: prune
                    e = path.pop(); x = self.to[e ^ 1]
                    it[x] += 1
            for e in path:
                self.cap[e] -= 1; self.cap[e ^ 1] += 1
            flow += 1
            trace.append([self.to[e ^ 1] for e in path] + [t])

    def run(self):
        total, phases, traces, phase_len = 0, 0, [], []
        while True:
            level = self.bfs()
            if level[self.t] < 0:
                break                              # no augmenting path  (Berge)
            phases += 1
            phase_len.append(level[self.t])
            tr = []
            total += self.blocking_flow(level, tr)
            traces.append(tr)
        return total, phases, phase_len, traces

    def extract(self):
        match = [-1] * self.U
        for (u, v), e in self.uv_edge.items():
            if self.cap[e] == 0:
                match[u] = v
        return match

    def reachable_from_source(self):
        seen = {self.s}
        q = deque([self.s])
        while q:
            x = q.popleft()
            for e in self.adj[x]:
                if self.cap[e] > 0 and self.to[e] not in seen:
                    seen.add(self.to[e]); q.append(self.to[e])
        return seen


def max_matching(cand_adj, caps, warm=None):
    net = Network(cand_adj, caps)
    if warm is not None:
        net.push_initial(cand_adj, warm)
    base = sum(1 for v in (warm or []) if v >= 0)
    gained, phases, phase_len, traces = net.run()
    return net, base + gained, phases, phase_len, traces


# ----------------------------------------------------------------------------- ILP benchmark
def ilp_solve(cand_adj, caps, relax=False):
    from scipy.optimize import milp, linprog, LinearConstraint, Bounds
    from scipy.sparse import coo_matrix
    edges = [(u, v) for u, nb in enumerate(cand_adj) for v in nb]
    m = len(edges)
    if m == 0:
        return 0, 0.0
    rows, cols = [], []
    for j, (u, v) in enumerate(edges):
        rows += [u, len(cand_adj) + v]; cols += [j, j]
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(cand_adj) + len(caps), m)).tocsr()
    ub = np.array([1] * len(cand_adj) + list(caps), dtype=float)
    c = -np.ones(m)
    t0 = time.perf_counter()
    if relax:
        res = linprog(c, A_ub=A, b_ub=ub, bounds=(0, 1), method="highs")
        val = -res.fun
    else:
        res = milp(c, constraints=LinearConstraint(A, -np.inf, ub), integrality=np.ones(m), bounds=Bounds(0, 1))
        val = -res.fun
    return int(round(val)), time.perf_counter() - t0


# ----------------------------------------------------------------------------- helpers
def verify_feasible(cand_adj, caps, match):
    load = [0] * len(caps)
    for u, v in enumerate(match):
        if v >= 0:
            assert v in cand_adj[u], "assigned to an ineligible role"
            load[v] += 1
    assert all(load[v] <= caps[v] for v in range(len(caps))), "capacity violated"
    return load


def reassignment_chains(cand_adj, match_greedy, match_final):
    """Candidates whose role changed (or who got newly hired) between greedy and optimal."""
    moved = [(u, match_greedy[u], match_final[u]) for u in range(len(cand_adj))
             if match_greedy[u] != match_final[u]]
    return moved


def draw(cand_adj, caps, mg, mf, traces, out, cand_ids, role_ids, n_chains=3):
    """Left panel: the longest reassignment chains, each in its own band, nodes ordered along the
    path. Red dashed = greedy edge that gets undone, green = edge in the optimal matching.
    Right panel: greedy vs optimal vs capacity."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    U = len(cand_adj)
    long_paths = sorted([p for tr in traces for p in tr if len(p) > 4], key=len, reverse=True)[:n_chains]
    G, pos, cand_nodes, role_nodes, final_e, undone_e = nx.Graph(), {}, [], [], [], []
    row = 0.0
    for k, p in enumerate(long_paths):
        nodes = p[1:-1]
        cs = [x - 1 for x in nodes if 1 <= x <= U]
        rs = [x - 1 - U for x in nodes if x > U]
        for i, u in enumerate(cs):
            cn = ("c", k, u); cand_nodes.append(cn); pos[cn] = (0, -(row + i))
        for i, v in enumerate(rs):
            rn = ("r", k, v); role_nodes.append(rn); pos[rn] = (1, -(row + i))
        for u in cs:
            for tag, m, bucket in (("f", mf, final_e), ("g", mg, undone_e)):
                v = m[u]
                if v >= 0 and ("r", k, v) in pos and not (tag == "g" and mg[u] == mf[u]):
                    bucket.append((("c", k, u), ("r", k, v)))
        row += len(cs) + 1.2
    G.add_nodes_from(cand_nodes + role_nodes)
    G.add_edges_from(final_e + undone_e)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 8), gridspec_kw={"width_ratios": [2.0, 1]})
    nx.draw_networkx_edges(G, pos, edgelist=undone_e, edge_color="#d62728", style="dashed", width=1.8, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=final_e, edge_color="#2ca02c", width=2.6, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=cand_nodes, node_color="#9ecae1", node_size=520, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=role_nodes, node_color="#fdd0a2", node_size=900, node_shape="s", ax=ax)
    nx.draw_networkx_labels(G, pos, {n: str(cand_ids[n[2]]) for n in cand_nodes}, font_size=7, ax=ax)
    nx.draw_networkx_labels(G, pos, {n: f"{role_ids[n[2]]}\ncap {caps[n[2]]}" for n in role_nodes}, font_size=6, ax=ax)
    ax.plot([], [], color="#2ca02c", lw=2.6, label="edge in optimal matching")
    ax.plot([], [], color="#d62728", ls="--", lw=1.8, label="greedy edge undone")
    ax.legend(loc="lower center", ncol=2, fontsize=9, frameon=False)
    ax.set_title(f"{len(long_paths)} longest augmenting chains (one band each): candidates | roles" if long_paths else "No reassignment chains were needed (greedy was already optimal)")
    ax.set_xlim(-0.15, 1.15); ax.axis("off")

    hg, hf, tot = sum(v >= 0 for v in mg), sum(v >= 0 for v in mf), sum(caps)
    bars = bx.bar(["Greedy", "Hopcroft-Karp\n(optimal)", "Total role\ncapacity"], [hg, hf, tot],
                  color=["#d62728", "#2ca02c", "#999999"])
    for b in bars:
        bx.text(b.get_x() + b.get_width() / 2, b.get_height() + 10, str(int(b.get_height())), ha="center")
    bx.set_title(f"Hires: greedy capacity gap = {hf - hg}")
    bx.set_ylim(0, max(hf, tot) * 1.12)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description="Maximum bipartite matching with role capacities.")
    ap.add_argument("--edges", help="csv: candidate_id,role_id (row order = preference order)")
    ap.add_argument("--roles", help="csv: role_id,capacity")
    ap.add_argument("--candidates", help="optional csv: candidate_id (keeps candidates with no eligible role)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--no-bench", action="store_true", help="skip the scipy ILP/LP and scaling benchmark")
    ap.add_argument("--outdir", default=".", help="where to write outputs (default: current directory)")
    a = ap.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)
    out = lambda name: os.path.join(a.outdir, name)

    try:
        if a.edges or a.roles:
            if not (a.edges and a.roles):
                raise ValueError("--edges and --roles must be given together")
            cand_adj, caps, cand_ids, role_ids = load_csv(a.edges, a.roles, a.candidates)
        else:
            cand_adj, caps = generate(a.seed)
            cand_ids = [f"C{u:04d}" for u in range(len(cand_adj))]
            role_ids = [f"R{v:03d}" for v in range(len(caps))]
            with open(out("data_roles.csv"), "w", newline="") as f:
                w = csv.writer(f); w.writerow(["role_id", "capacity"])
                for v, c in enumerate(caps): w.writerow([role_ids[v], c])
            with open(out("data_edges.csv"), "w", newline="") as f:
                w = csv.writer(f); w.writerow(["candidate_id", "role_id"])
                for u, nb in enumerate(cand_adj):
                    for v in nb: w.writerow([cand_ids[u], role_ids[v]])
    except (ValueError, OSError) as ex:
        print(f"input error: {ex}", file=sys.stderr)
        return 2

    U, V, E = len(cand_adj), len(caps), sum(len(x) for x in cand_adj)
    R = {"candidates": U, "roles": V, "edges": E, "total_capacity": sum(caps)}

    t0 = time.perf_counter(); mg = greedy(cand_adj, caps); R["t_greedy"] = time.perf_counter() - t0
    R["greedy_hires"] = sum(v >= 0 for v in mg)

    # cold start (benchmark) and warm start from greedy (path traces)
    t0 = time.perf_counter(); _, cold, cold_phases, _, _ = max_matching(cand_adj, caps)
    R["t_hk_cold"] = time.perf_counter() - t0; R["hk_cold_hires"] = cold; R["hk_cold_phases"] = cold_phases

    t0 = time.perf_counter()
    net, opt, phases, phase_len, traces = max_matching(cand_adj, caps, warm=mg)
    R["t_hk_warm"] = time.perf_counter() - t0
    mf = net.extract()
    verify_feasible(cand_adj, caps, mf)
    assert sum(v >= 0 for v in mf) == opt == cold, "warm/cold disagree"
    R["optimal_hires"] = opt; R["greedy_gap"] = opt - R["greedy_hires"]
    R["warm_phases"] = phases; R["warm_phase_path_len"] = phase_len
    paths = [p for tr in traces for p in tr]
    R["augmenting_paths"] = len(paths)
    R["paths_len3"] = sum(len(p) == 4 for p in paths)         # s,u,v,t : plain new hire
    R["paths_reassigning"] = sum(len(p) > 4 for p in paths)   # chains undoing greedy choices
    R["max_chain_len_edges"] = max((len(p) - 1 for p in paths), default=0)
    R["candidates_moved"] = len([1 for u in range(U) if mg[u] >= 0 and mf[u] != mg[u]])

    # Berge certificate: t unreachable in the residual graph; the reachable side gives a min cut
    reach = net.reachable_from_source()
    R["berge_no_augmenting_path"] = net.t not in reach
    cut = 0
    for e in range(0, len(net.to), 2):
        if net.to[e ^ 1] in reach and net.to[e] not in reach:
            cut += net.orig_cap[e]
    R["min_cut_capacity"] = cut
    assert cut == opt, "min-cut must equal max-flow"

    # networkx validation (flow value only; the engine itself is hand-written)
    if E > 0:
        import networkx as nx
        G = nx.DiGraph()
        for u, nb in enumerate(cand_adj):
            G.add_edge("s", ("c", u), capacity=1)
            for v in nb: G.add_edge(("c", u), ("r", v), capacity=1)
        for v, c in enumerate(caps): G.add_edge(("r", v), "t", capacity=c)
        R["networkx_max_flow"] = nx.maximum_flow_value(G, "s", "t") if "t" in G else 0
    else:
        R["networkx_max_flow"] = 0
    assert R["networkx_max_flow"] == opt, "disagrees with networkx"

    # ILP / LP benchmark
    if not a.no_bench:
        R["ilp_hires"], R["t_ilp"] = ilp_solve(cand_adj, caps)
        R["lp_hires"], R["t_lp"] = ilp_solve(cand_adj, caps, relax=True)
        assert R["ilp_hires"] == opt == R["lp_hires"], "ILP/LP disagree with engine"
        scal = []
        for k in (1, 2, 4, 8):
            ca, cp = generate(a.seed + k, 1200 * k, 160 * k)
            Ek = sum(len(x) for x in ca)
            t0 = time.perf_counter(); _, hires, _, _, _ = max_matching(ca, cp); th = time.perf_counter() - t0
            ih, ti = ilp_solve(ca, cp)
            assert ih == hires
            scal.append({"U": len(ca), "V": len(cp), "E": Ek, "hires": hires, "t_hk": th, "t_ilp": ti})
        R["scaling"] = scal

    # deliverable: assignment table (original ids preserved)
    with open(out("assignment.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "role_id", "status", "greedy_role_id", "changed_vs_greedy"])
        for u in range(U):
            g = role_ids[mg[u]] if mg[u] >= 0 else ""
            r = role_ids[mf[u]] if mf[u] >= 0 else ""
            w.writerow([cand_ids[u], r, "ASSIGNED" if mf[u] >= 0 else "UNASSIGNED", g, int(mg[u] != mf[u])])

    # human-readable traces of the longest reassignment chains
    longest = sorted([p for p in paths if len(p) > 4], key=len, reverse=True)[:5]
    R["example_traces"] = []
    for p in longest:
        nodes, txt = p[1:-1], []
        for i in range(len(nodes) - 1):
            x, y = nodes[i], nodes[i + 1]
            if 1 <= x <= U:      # candidate -> role
                txt.append(f"{cand_ids[x-1]} takes {role_ids[y-1-U]}")
            else:                # role -> candidate (residual back edge): candidate leaves that role
                txt.append(f"({cand_ids[y-1]} vacates {role_ids[x-1-U]})")
        R["example_traces"].append({"edges": len(p) - 1, "steps": txt})

    draw(cand_adj, caps, mg, mf, traces, out("matching_vis.png"), cand_ids, role_ids)
    with open(out("results.json"), "w") as f:
        json.dump(R, f, indent=2)
    print(json.dumps({k: v for k, v in R.items() if k not in ("example_traces", "scaling")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
