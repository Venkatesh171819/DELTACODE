# Delta Code 2026 – PS 1: Maximum Bipartite Matching via Augmenting Paths

## 1. Setup

No dataset ships with the problem statement, so `allocate.py` generates a seeded synthetic workforce dataset (seed 2026) and also accepts real CSVs (`--edges edges.csv --roles roles.csv`). The generated data is saved as `data_edges.csv` / `data_roles.csv`.

| Quantity | Value |
|---|---|
| Candidates \|U\| | 2,400 |
| Roles \|V\| | 320 |
| Eligibility edges \|E\| | 6,142 |
| Total capacity Σc_v | 2,039 |

Each candidate is eligible for 1–5 roles (listed in preference order). Role popularity is skewed (a few "hot" roles), and capacities are 3–10. This mimics the setting where first-fit greedy locks hot roles early.

**Flow network G′.** Source s, sink t. Edges (s,u) with capacity 1, (u,v) with capacity 1 for each eligible pair, and (v,t) with capacity c_v. Integral max-flow = maximum hires.

**Engine (hand-written).** Hopcroft–Karp in its Dinic form: BFS builds the level graph, an iterative DFS with current-arc pointers finds a blocking set of shortest augmenting paths, repeated until BFS cannot reach t. networkx is used only for flow validation and drawing; scipy only for the ILP/LP benchmark.

**Greedy baseline.** Candidates in ID order take their first open eligible role (irreversible). The engine is then warm-started from this greedy matching, so every augmenting path it finds is exactly a reassignment chain that repairs a greedy mistake.

## 2. Results

| Method | Hires |
|---|---|
| Greedy first-fit | 1,709 |
| Hopcroft–Karp (optimal) | **1,901** |
| networkx max-flow (validation) | 1,901 |
| scipy `milp` (ILP) | 1,901 |
| scipy `linprog` (LP relaxation) | 1,901 |

**Greedy Capacity Gap = 1,901 − 1,709 = 192 hires** (11.2% more hires than greedy; 10.1% of the optimum). 499 candidates remain unassigned in the optimum, and 138 of the 2,039 slots stay empty because no eligible unassigned candidate can reach them (see the cut certificate in §3).

All 192 augmenting paths found from the greedy start were multi-step reassignment chains (none was a plain s→u→v→t path, since greedy is already maximal in the "free slot" sense). 251 candidates ended up in a different role than greedy gave them.

## 3. Berge's Lemma and the optimality proof

**Definitions.** For a matching M in a graph, an *M-augmenting path* is a path that alternates between non-M and M edges and starts and ends at M-unmatched vertices. To handle capacities, replace each role v by c_v copies ("slots"), each adjacent to the same candidates. Matchings in the expanded graph correspond exactly to feasible assignments, and the maximum matching size equals the max flow in G′.

**Lemma (Berge).** M is a maximum matching ⇔ there is no M-augmenting path.

*Proof.*
(⇒) Suppose an augmenting path P = v₀v₁…v_k exists. It has k odd, with (k+1)/2 non-M edges and (k−1)/2 M edges. Then M′ = M △ E(P) is a matching (the endpoints were unmatched, interior vertices keep exactly one incident M′ edge) with |M′| = |M| + 1, so M is not maximum.

(⇐) Suppose M is not maximum; take a matching M′ with |M′| > |M|. Consider H = M △ M′. Every vertex has degree ≤ 2 in H (at most one edge from each matching), so each component is a path or an even cycle, and in each the edges alternate between M and M′. Even cycles and even-length paths contain equally many M and M′ edges. Since |M′| > |M|, some component has more M′ edges than M edges, so it is a path of odd length that begins and ends with M′ edges. Each endpoint x has degree 1 in H, and its single H-edge is an M′ edge (so not in M). If M covered x by some other edge e, then e ∉ M′ (M′ has only one edge at x), so e ∈ H and x would have degree 2 in H, a contradiction. So both endpoints are M-unmatched. Hence it is an M-augmenting path. ∎

**Flow form.** In G′, a path s→…→t in the residual graph is exactly an augmenting path: forward edges u→v are "assign", backward edges v→u′ are "u′ vacates v". Pushing one unit flips the edge states along the path and raises |M| by exactly 1.

**Certificate on this data.** After the last phase, residual BFS from s does not reach t (`berge_no_augmenting_path = True`). The set S of vertices reachable from s defines an s–t cut whose original capacity is **1,901**, equal to the flow value, which proves optimality independently (max-flow/min-cut). This agrees with networkx, ILP and LP.

## 4. Complexity: O(|E|√|V|)

*Cost of one phase = O(|E|).* BFS is O(|E|). In the blocking-flow DFS, the current-arc pointers make every edge a dead end at most once per phase. Every edge except the (v,t) edges has capacity 1, so it is saturated, and used by at most one path, per phase; each path has exactly one (v,t) edge. Total path work is therefore at most |E| + (#paths) = O(|E|).

*Number of phases = O(√N), N = |U|+|V|+2.* The shortest augmenting-path length strictly increases after every phase. After √N phases it is ≥ √N. The remaining flow (M* − M) decomposes into augmenting paths of that length. Each candidate vertex has unit throughput (its only in-edge is (s,u), capacity 1), so these paths are candidate-disjoint, and each contains ≥ √N/2 candidates. There are therefore at most 2|U|/√N ≤ 2√N of them, and each later phase adds at least one. Hence O(√N) phases and total time **O(|E|√|V|)** (with |V| read as the vertex count of G′).

*Observed:* only **4 phases** from the greedy start (shortest-path lengths 5, 7, 9, 11 edges), and 5 phases from scratch, far below the √N ≈ 52 worst-case scale.

## 5. Path traces: how greedy misallocations were undone

Each trace reads left to right; "(X vacates R)" is a backward edge in the residual graph, i.e. X gives up the slot greedy had given them.

**Chain 1 (11 edges).** C1173 takes R219 → (C0540 vacates R219) → C0540 takes R300 → (C0768 vacates R300) → C0768 takes R060 → (C0700 vacates R060) → C0700 takes R277 → (C2215 vacates R277) → C2215 takes R017 (a slot with free capacity).
Interpretation: C1173 had no open role because greedy filled R219 with C0540 early. Shifting four earlier hires one step along their alternative eligible roles frees a seat at R219 and ends at R017, which still had room. Net +1 hire; no one loses a job.

**Chain 2 (11 edges).** C1260 takes R073 → (C0011 vacates R073) → C0011 takes R306 → (C0777 vacates R306) → C0777 takes R192 → (C2043 vacates R192) → C2043 takes R221 → (C1831 vacates R221) → C1831 takes R295.

**Chain 3 (11 edges).** C1280 takes R064 → (C0714 vacates R064) → C0714 takes R261 → (C0949 vacates R261) → C0949 takes R279 → (C1412 vacates R279) → C1412 takes R171 → (C1689 vacates R171) → C1689 takes R223.

Per phase the shortest augmenting paths grow (5 → 7 → 9 → 11 edges), matching the theory: short fixes are exhausted first, then longer chains are needed. `matching_vis.png` draws the three longest chains, one band each (the staircase of red dashed → green edges is the chain): green = final matching edge, red dashed = greedy edge that was undone, grey = eligible but unused; the right panel shows greedy vs optimal vs total capacity.

## 6. Empirical benchmark

Single run on the main instance (CPU, one thread):

| Method | Time |
|---|---|
| Greedy | 0.3 ms |
| Hopcroft–Karp, warm start | 25 ms |
| Hopcroft–Karp, from scratch | 30 ms |
| scipy `linprog` (LP, HiGHS) | 35 ms |
| scipy `milp` (ILP, HiGHS) | 123 ms |

The LP relaxation is already integral (the constraint matrix is totally unimodular), so LP, ILP and flow agree.

Scaling (instances scaled proportionally, same generator, new seeds; `milp` result asserted equal to HK):

| \|U\| | \|V\| | \|E\| | Hires | HK (ms) | ILP (ms) | ILP/HK |
|---|---|---|---|---|---|---|
| 1,200 | 160 | 3,053 | 938 | 19 | 73 | 3.8× |
| 2,400 | 320 | 6,101 | 1,856 | 31 | 102 | 3.3× |
| 4,800 | 640 | 12,236 | 3,619 | 75 | 295 | 3.9× |
| 9,600 | 1,280 | 24,732 | 7,219 | 230 | 842 | 3.7× |

Doubling the size three times (|E| ×8.1) increased HK time ×12, i.e. roughly |E|^1.19, comfortably inside the O(|E|√|V|) bound (which would allow ≈×23). The ILP stays about 3–4× slower at every size and offers no better answer here. (Timings vary a few percent from run to run; `results.json` holds the latest.)

## 7. Takeaways and caveats

1. First-fit greedy leaves 192 hires (11%) on the table on this instance; the loss comes from irreversible choices on hot roles.
2. Augmenting-path repair recovers all of it with a handful of long reassignment chains and no one being fired.
3. Optimality is certified three ways: Berge (no augmenting path), a matching min-cut of 1,901, and agreement with networkx, LP and ILP.
4. The dataset is synthetic. The size of the gap depends on how skewed popularity and preference orders are; rerun with real data via `--edges/--roles`.

## 8. Correctness, input validation and edge cases

`test_allocate.py` (8 tests, `python test_allocate.py`):

| Test | What it checks |
|---|---|
| random small ×300 | legal assignment, equals networkx max-flow, no augmenting path left, greedy ≤ optimal (cold and warm start) |
| main instance | 1,901 hires, feasible |
| real ids / duplicates / zero capacity | original ids (e.g. `alice`, `JobA`) survive into `assignment.csv`; duplicate edges collapsed; a 0-seat role never receives anyone |
| candidate without edges | kept as `UNASSIGNED` when passed via `--candidates` |
| more candidates than seats | exactly the seat count is hired |
| empty edge list | runs cleanly, empty assignment |
| bad inputs | unknown role id, negative or non-integer capacity, duplicate role id, missing column and empty id each stop with a readable message and exit code 2 |
| 40-step forced chain | a long augmenting path is found and the result is optimal |

Every run also asserts: assignments are eligible, no role exceeds capacity, warm = cold result, min-cut = max-flow = networkx = ILP = LP.

## 9. Reproduce

```
pip install -r requirements.txt
python test_allocate.py       # correctness tests (300 random instances vs networkx)
python allocate.py            # writes assignment.csv, matching_vis.png, results.json, data_*.csv
python allocate.py --edges edges.csv --roles roles.csv [--candidates candidates.csv] [--outdir out]   # real data
```
Requires numpy, scipy, networkx, matplotlib.
