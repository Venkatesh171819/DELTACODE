"""Correctness tests: python test_allocate.py  (no pytest needed)"""
import csv
import os
import random
import subprocess
import sys
import tempfile
import networkx as nx
from allocate import generate, greedy, max_matching, verify_feasible, load_csv

HERE = os.path.dirname(os.path.abspath(__file__))


def nx_flow(cand_adj, caps):
    G = nx.DiGraph()
    for u, nb in enumerate(cand_adj):
        G.add_edge("s", ("c", u), capacity=1)
        for v in nb: G.add_edge(("c", u), ("r", v), capacity=1)
    for v, c in enumerate(caps): G.add_edge(("r", v), "t", capacity=c)
    return nx.maximum_flow_value(G, "s", "t")


def test_random_small():
    rng = random.Random(1)
    for _ in range(300):
        U, V = rng.randint(1, 25), rng.randint(1, 8)
        caps = [rng.randint(1, 4) for _ in range(V)]
        adj = [rng.sample(range(V), rng.randint(0, min(V, 4))) for _ in range(U)]
        mg = greedy(adj, caps)
        for warm in (None, mg):
            net, opt, *_ = max_matching(adj, caps, warm=warm)
            m = net.extract()
            verify_feasible(adj, caps, m)
            assert sum(v >= 0 for v in m) == opt == nx_flow(adj, caps)
            assert net.t not in net.reachable_from_source()      # Berge: no augmenting path left
        assert sum(v >= 0 for v in mg) <= opt


def test_main_instance():
    adj, caps = generate()
    net, opt, *_ = max_matching(adj, caps, warm=greedy(adj, caps))
    assert opt == nx_flow(adj, caps) == 1901
    verify_feasible(adj, caps, net.extract())


def cli(edges, roles, candidates=None):
    """Run allocate.py on in-memory csv text; return (exit_code, stderr, assignment rows or None)."""
    with tempfile.TemporaryDirectory() as d:
        for name, txt in (("e.csv", edges), ("r.csv", roles), ("c.csv", candidates)):
            if txt is not None:
                open(os.path.join(d, name), "w").write(txt)
        cmd = [sys.executable, os.path.join(HERE, "allocate.py"), "--edges", os.path.join(d, "e.csv"),
               "--roles", os.path.join(d, "r.csv"), "--no-bench", "--outdir", os.path.join(d, "out")]
        if candidates is not None:
            cmd += ["--candidates", os.path.join(d, "c.csv")]
        p = subprocess.run(cmd, capture_output=True, text=True)
        rows = None
        f = os.path.join(d, "out", "assignment.csv")
        if os.path.exists(f):
            rows = {r["candidate_id"]: r for r in csv.DictReader(open(f))}
        return p.returncode, p.stderr, rows


def test_real_ids_duplicates_zero_capacity():
    code, err, rows = cli("candidate_id,role_id\nalice,JobA\nalice,JobA\nbob,JobA\nbob,JobB\ncara,JobC\n",
                          "role_id,capacity\nJobA,1\nJobB,0\nJobC,2\n")
    assert code == 0, err
    assert set(rows) == {"alice", "bob", "cara"}                       # original ids preserved
    assert rows["alice"]["role_id"] == "JobA" and rows["cara"]["role_id"] == "JobC"
    assert rows["bob"]["status"] == "UNASSIGNED"                       # JobB has zero seats, JobA is taken


def test_candidate_without_edges_is_kept():
    code, err, rows = cli("candidate_id,role_id\nalice,JobA\n", "role_id,capacity\nJobA,1\n", "candidate_id\nalice\nzed\n")
    assert code == 0, err
    assert rows["zed"]["status"] == "UNASSIGNED" and rows["alice"]["status"] == "ASSIGNED"


def test_more_candidates_than_seats():
    edges = "candidate_id,role_id\n" + "".join(f"c{i},J\n" for i in range(7))
    code, err, rows = cli(edges, "role_id,capacity\nJ,3\n")
    assert code == 0, err
    assert sum(r["status"] == "ASSIGNED" for r in rows.values()) == 3


def test_empty_edges():
    code, err, rows = cli("candidate_id,role_id\n", "role_id,capacity\nJ,2\n")
    assert code == 0, err
    assert rows == {}


def test_bad_inputs_give_clear_errors():
    ok_roles = "role_id,capacity\nJ,1\n"
    for edges, roles, msg in [
        ("candidate_id,role_id\na,ZZZ\n", ok_roles, "not listed"),             # unknown role
        ("candidate_id,role_id\na,J\n", "role_id,capacity\nJ,-1\n", "negative"),
        ("candidate_id,role_id\na,J\n", "role_id,capacity\nJ,two\n", "not an integer"),
        ("candidate_id,role_id\na,J\n", "role_id,capacity\nJ,1\nJ,2\n", "duplicate"),
        ("who,role_id\na,J\n", ok_roles, "missing column"),
        ("candidate_id,role_id\n,J\n", ok_roles, "empty"),
    ]:
        code, err, _ = cli(edges, roles)
        assert code == 2 and msg in err, (msg, code, err)


def test_long_chain_instance():
    """Candidate 0 greedily takes role 0, so the last candidate (who only fits role 0) is blocked.
    The only repair is a long chain: 0 -> role 1, 1 -> role 2, ..., pushing everyone along one seat."""
    n = 40
    adj = [[0, 1]] + [[i, i + 1] for i in range(1, n - 1)] + [[0]]
    caps = [1] * n
    mg = greedy(adj, caps)
    net, opt, phases, lens, traces = max_matching(adj, caps, warm=mg)
    assert opt == nx_flow(adj, caps) > sum(v >= 0 for v in mg)
    assert max(len(p) for tr in traces for p in tr) > 8                  # genuinely long chain
    verify_feasible(adj, caps, net.extract())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all tests passed")
    raise SystemExit(0)

