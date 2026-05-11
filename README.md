# Multi-Objective RL for Carbon-Aware Global Logistics

**CSCI-6364 S26 — Machine Learning**
**Team:** Alekya Kowta · Kushi Khandoji · Kritika Berry · Jasmine Sciarillo

A routing engine for freight logistics that jointly optimises **delivery time** and **CO₂ emissions** on a real Washington DC road network. The graph is enriched with MOVES5 emission factors, SRTM elevation grades, and 4 time-of-day congestion profiles. Three algorithms are compared across an α ∈ [0,1] preference sweep (α=1 = time only, α=0 = carbon only).

---

## Results Summary

### Weighted A* — exact optimal (Phase 3)
- **100% routing success** on all 20 evaluation pairs
- At α=0.3 (sweet spot): **−3.1% carbon** for only **+1.7% travel time** vs fastest route
- Produces a smooth, provably optimal Pareto curve

### PPO + Behavioral Cloning — policy-based RL (Phase 2)
- 5 policies trained, each with its own Pareto-weighted Dijkstra expert
- Greenest policy achieves **0.946 carbon optimality** (94.6% of theoretical best)
- Pareto ordering correct: greenest = lowest carbon, time-biased = fastest

### Preference-conditioned DQN — value-based RL (Phase 3)
- Single network conditioned on α — handles all preferences without retraining
- 2–15% routing success after 120,000 training steps (needs more training)
- Action-as-input architecture fixes the generalisation problem of fixed-head DQN

### Terrain study
- **San Francisco (hilly) has 85% more carbon-aware routing benefit than DC (flat)**
- DC: 2.14% mean savings · SF: 3.97% mean savings

---

## Repository Structure

```
.
├── README.md
├── REPORT_2.md                          <- full project report (all results)
├── SPECIFICATIONS.md                    <- detailed architecture doc
├── requirements.txt
│
├── src/                                 <- Phase 1 data pipeline + Phase 2 PPO
│   ├── run_pipeline.py                  <- ONE COMMAND: runs all 4 pipeline stages
│   ├── extract_network.py               <- OSM road network extraction
│   ├── elevation_data.py                <- SRTM 30m elevation + road grade
│   ├── add_speeds.py                    <- 4 time-of-day speed profiles per edge
│   ├── carbon_calculate.py              <- MOVES5 -> 12 carbon weights per edge
│   ├── rl_env.py                        <- Gymnasium MDP environment (154-dim state)
│   ├── pretrain_bc.py                   <- per-policy Behavioral Cloning training
│   ├── train_ppo.py                     <- optional PPO fine-tuning on BC models
│   ├── evaluate.py                      <- PPO evaluation + Pareto plots
│   └── terrain_study.py                 <- DC vs SF terrain comparison
│
├── baseline/                            <- Phase 3 model comparison
│   ├── single_file_morl.py              <- Weighted A* + DQN + Dijkstra (main)
│   ├── weighted_dijkstra.py             <- Weighted Dijkstra per-trip sweep
│   └── pareto_front_dijkstra.png        <- pre-generated per-trip scatter
│
├── data/
│   ├── dc_subgraph_carbon.graphml       <- MAIN: ~960-node DC graph, 12 carbon weights/edge
│   ├── terrain_dc_carbon.graphml        <- DC graph for terrain study (1432 nodes)
│   ├── terrain_sf_carbon.graphml        <- SF graph for terrain study (1477 nodes)
│   └── carbon_heatmap.png               <- carbon intensity visualisation
│
├── models/
│   ├── ppo_fastest_bc.zip               <- PPO policy α=1.00 β=0.00
│   ├── ppo_time_biased_bc.zip           <- PPO policy α=0.75 β=0.25
│   ├── ppo_balanced_bc.zip              <- PPO policy α=0.50 β=0.50
│   ├── ppo_carbon_biased_bc.zip         <- PPO policy α=0.25 β=0.75
│   ├── ppo_greenest_bc.zip              <- PPO policy α=0.00 β=1.00
│   └── dqn_carbon.pt                    <- preference-conditioned DQN checkpoint
│
└── results/
    ├── pareto_frontier_astar_dqn.png    <- A* smooth curve + DQN scatter  [Phase 3 main]
    ├── pareto_front_dijkstra.png        <- per-trip weighted Dijkstra scatter
    ├── pareto_frontier_ppo.png          <- PPO 5-policy Pareto frontier    [Phase 2 main]
    ├── policy_metrics_ppo.png           <- PPO carbon optimality bar chart
    ├── terrain_comparison.png           <- DC vs SF terrain validation
    ├── summary_astar_dqn.csv            <- A*/DQN per-alpha results table
    ├── evaluation_metrics_ppo.csv       <- PPO per-trip data (50 trips x 5 policies)
    └── terrain_comparison.csv           <- terrain study summary
```

---

## Setup (once)

```bash
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

---

## Phase 1 — Build the Data Pipeline

Converts OpenStreetMap road data into a routing graph with per-edge CO₂ weights.
**Skip this if you already have `data/dc_subgraph_carbon.graphml`** (it's included in the repo).

```bash
cd src
python3 run_pipeline.py         # all 4 stages in one command, ~5 min
```

What it builds:

| Stage | Script | What it does |
|---|---|---|
| 1 | `extract_network.py` | Pulls ~960-node DC subgraph from OpenStreetMap |
| 2 | `elevation_data.py` | Adds SRTM 30m elevation to every node; computes directed road grade per edge |
| 3 | `add_speeds.py` | 4 time-of-day speed multipliers: midnight(1.0x) lunch(0.85x) morning-rush(0.6x) evening-rush(0.5x) |
| 4 | `carbon_calculate.py` | MOVES5 emission curve × 3 vehicle classes × 4 time periods = **12 carbon weights per edge** |

---

## Phase 2 — PPO + Behavioral Cloning

Five MaskablePPO policies, each trained on a Dijkstra expert matched to its (α, β) preference.

### Step 1 — Train BC policies

```bash
cd src
python3 pretrain_bc.py                     # trains all 5 policies, ~2 min
python3 pretrain_bc.py --label balanced    # train one policy only
```

What it trains:

| Policy | α (time) | β (carbon) | Its expert |
|---|---|---|---|
| fastest | 1.00 | 0.00 | Dijkstra minimising time |
| time_biased | 0.75 | 0.25 | Dijkstra on 0.75·time + 0.25·carbon |
| balanced | 0.50 | 0.50 | Dijkstra on equal-weighted cost |
| carbon_biased | 0.25 | 0.75 | Dijkstra on 0.25·time + 0.75·carbon |
| greenest | 0.00 | 1.00 | Dijkstra minimising carbon |

Saves: `models/ppo_*_bc.zip`

### Step 2 — Evaluate PPO policies

```bash
cd src
python3 evaluate.py --no-cycle-break --max-detour 2.5
```

Outputs:
- `results/pareto_frontier_ppo.png` — 5-policy Pareto frontier
- `results/policy_metrics_ppo.png` — carbon optimality bar chart
- `results/evaluation_metrics_ppo.csv` — per-trip data

Expected results (50 trips, full DC graph, honest evaluation):

| Policy | Success | Avg Time | Avg CO₂ | Carbon Optimality |
|---|---|---|---|---|
| greenest (β=1.0) | 10/50 | 526.6 s | **2,035 g** | **0.946** ← best |
| fastest (α=1.0) | 10/50 | 568.4 s | 2,303 g | 0.933 |
| balanced | 15/50 | 586.3 s | 2,409 g | 0.907 |
| carbon_biased | 15/50 | 562.0 s | 2,279 g | 0.905 |
| time_biased (α=0.75) | 13/50 | **524.1 s** | 2,207 g | 0.894 |

Dijkstra baselines on same trips: Fastest = 469s/1885g · Greenest = 477s/1853g

### Step 3 — (Optional) PPO fine-tuning

```bash
cd src
python3 train_ppo.py --use-bc --phase1 100000 --phase2 200000 --phase3 300000
```

### Step 4 — (Optional) Terrain study: DC vs San Francisco

```bash
cd src
python3 terrain_study.py        # builds SF graph from scratch, ~5 min
```

Output: `results/terrain_comparison.png`

Result: SF (hilly) has **3.97%** mean carbon savings vs DC's **2.14%** — 85% more benefit.

---

## Phase 3 — Weighted A* + Preference DQN (baseline comparison)

### Option A — Weighted A* only (no training needed, ~10 sec)

The exact optimal Pareto solver. Use this to see the true achievable tradeoff.

```bash
python3 baseline/single_file_morl.py astar --pairs 20
```

Output: `results/pareto_frontier_astar_dqn.png`, `results/summary_astar_dqn.csv`

Expected results (20 OD pairs, 4–16 hops, 100% success):

| α | Avg time | Avg CO₂ | Time vs fastest | Carbon vs fastest |
|---|---|---|---|---|
| 0.00 | 235.5 s | 910.4 g | +3.0% | **−3.3%** |
| **0.30** | **232.5 s** | **913.2 g** | **+1.7%** | **−3.1%** ← sweet spot |
| 0.50 | 230.7 s | 917.1 g | +1.0% | −2.6% |
| 1.00 | 228.6 s | 941.9 g | 0.0% | 0.0% |

### Option B — Train the preference-conditioned DQN (~30–60 min)

```bash
python3 baseline/single_file_morl.py train --steps 120000
# Saves: models/dqn_carbon.pt
```

### Option C — Compare all: A* + DQN + Dijkstra

```bash
python3 baseline/single_file_morl.py compare --pairs 20
```

DQN results after 120k steps: 2–15% success rate. On matched trips, DQN routes are 2–35% worse than A* — more training needed.

### Option D — Weighted Dijkstra per-trip scatter

```bash
python3 baseline/weighted_dijkstra.py
# Output: results/pareto_front_dijkstra.png
```

---

## Algorithm Comparison

| | Weighted A* | PPO + BC | Preference DQN |
|---|---|---|---|
| **Success rate** | **100%** | 20–30% | 2–15% |
| **Carbon optimality** | **1.000** (exact) | 0.946 | Converging |
| **Best carbon saving** | **−3.34%** | Per-trip ~−5% | Varies |
| Training needed | **None** | ~2 min BC | ~30–60 min RL |
| Generalises to new graphs | No (needs re-run) | Yes (learned weights) | Yes (learned) |
| Eval pairs used | 20 (4–16 hops) | 50 (full graph) | 20 (4–16 hops) |

**Why both A* and RL?**
A* is the oracle — it always finds the optimal route but requires the full graph structure at query time. RL agents learn a routing policy that can potentially generalise to new graphs, new cities, or new preferences without re-running search. A* tells us what's possible; RL tries to learn it.

---

## Output Plots Guide

| File | Shows | Use in report |
|---|---|---|
| `data/carbon_heatmap.png` | DC road network — brighter = higher CO₂ edge | Methodology section |
| `results/pareto_frontier_astar_dqn.png` | **A* smooth Pareto curve + DQN points** | Phase 3 results — show first |
| `results/pareto_front_dijkstra.png` | Per-trip Dijkstra scatter (colour = α preference) | Phase 3 supplement |
| `results/pareto_frontier_ppo.png` | **PPO 5-policy Pareto frontier** | Phase 2 results |
| `results/policy_metrics_ppo.png` | Carbon optimality 0.894–0.946 bar chart | Phase 2 results |
| `results/terrain_comparison.png` | DC 2.14% vs SF 3.97% carbon savings | Terrain validation |

---

## Troubleshooting

**`ModuleNotFoundError: stable_baselines3`**
→ Run `source venv/bin/activate` first

**PPO evaluation shows 0/50 success**
→ `models/ppo_*_bc.zip` files must exist. Run `python3 src/pretrain_bc.py` first.

**DQN shows 0/N success**
→ Needs more training. Run: `python3 baseline/single_file_morl.py train --steps 400000`

**Elevation API slow / timeout**
→ OpenTopoData is rate-limited (100 nodes/request, 1 req/sec). The pipeline retries automatically. Takes ~1 min for 960 nodes.

**`embeddings_*.npy` not found**
→ Safe to ignore. Auto-generated by `rl_env.py` on first env load and cached to `data/`.
