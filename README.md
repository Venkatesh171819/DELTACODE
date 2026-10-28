# Delta Code 2026 – PS 1: Maximum Bipartite Matching via Augmenting Paths

Hand-written Hopcroft–Karp (Dinic form) for capacitated bipartite matching, with a greedy baseline,
Berge optimality certificate, and scipy ILP/LP benchmark. networkx is used only for visualization
and flow validation.

| File | Purpose |
|---|---|
| `allocate.py` | matching engine, greedy baseline, benchmarks, writes all outputs |
| `assignment.csv` | candidate → role table (optimal, 1,901 hires) |
| `matching_vis.png` | bipartite rendering of the longest augmenting chains + greedy-vs-optimal bars |
| `report.md` | Berge proof, O(\|E\|√\|V\|) analysis, path traces, benchmarks |
| `test_allocate.py` | 8 tests: 300 random instances vs networkx, real ids, validation errors, edge cases |
| `Delta_Code_PS1_Deck.pptx` / `.pdf` | presentation: formulation, algorithm, results, performance, trade-offs |
| `ps1_notes.pdf` (+ `.tex`, `lindrewlite.sty`) | plain-language theory and code walkthrough |
| `data_edges.csv`, `data_roles.csv` | generated input (seed 2026) |
| `results.json` | all metrics from the last run |

```
pip install -r requirements.txt
python test_allocate.py
python allocate.py                                       # synthetic data
python allocate.py --edges edges.csv --roles roles.csv   # your own data
# optional: --candidates candidates.csv (keeps candidates with no eligible role), --outdir out, --no-bench
```
Inputs: `edges.csv` = `candidate_id,role_id` (row order = preference order); `roles.csv` = `role_id,capacity`.
