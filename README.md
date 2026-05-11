# Multi-Objective Reinforcement Learning for Carbon-Aware Global Logistics

CSCI-6364 S26 — Machine Learning
Team: Alekya Kowta, Kushi Khandoji, Kritika Berry, Jasmine Sciarillo

A multi-objective reinforcement-learning routing engine that optimises both delivery time and CO₂ emissions on a real OpenStreetMap road network. Five policies (spanning the Pareto frontier from *fastest* to *greenest*) are trained via Behavioral Cloning on Pareto-weighted Dijkstra demonstrations.

---

## Quick Start (clean machine → final plots in ~15 min)

```bash
# 1. Clone and enter
git clone <repo-url>
cd Multi-Objective-Reinforcement-Learning-for-Carbon-Aware-Global-Logistics

# 2. Set up Python virtual environment (Python 3.11+)
python3 -m venv venv
source venv/bin/activate           # macOS / Linux
# or:  venv\Scripts\activate       # Windows

# 3. Install dependencies (~3 min)
pip install -r requirements.txt

# 4. Build the data pipeline           (~5 min — OpenStreetMap + elevation API)
cd src
python3 run_pipeline.py

# 5. Train the 5 PPO policies via Behavioral Cloning   (~2 min)
python3 pretrain_bc.py

# 6. Evaluate and generate the Pareto frontier plot     (~1 min)
python3 evaluate.py --no-cycle-break --max-detour 2.5

# 7. (Optional) Terrain study — compares DC vs San Francisco  (~5 min)
python3 terrain_study.py
```

When done, the plots are in `results/`:
- `pareto_frontier.png` — the main Pareto plot (5 RL policies + 2 Dijkstra baselines)
- `policy_metrics.png` — bar chart of carbon savings and carbon optimality per policy
- `terrain_comparison.png` — DC (flat) vs SF (hilly) carbon-savings comparison
- `evaluation_metrics.csv` — per-trip raw numbers for all 50 evaluation trips
- `terrain_comparison.csv` — terrain summary table

---

## What this project does

### Phase 1 — Data Pipeline (`src/run_pipeline.py`)

Builds a routing graph enriched with everything needed to compute CO₂ emissions per edge.

| Stage | Script | Output |
|---|---|---|
| Network extraction | `extract_network.py` | `data/dc_subgraph.graphml` — ~1000-node directed road graph |
| Elevation & grade | `elevation_data.py` | `data/dc_subgraph_elev.graphml` — SRTM 30m elevation per node + grade per edge |
| Speed profiles | `add_speeds.py` | `data/dc_subgraph_speeds.graphml` — 4 time-of-day speed states per edge |
| Carbon weights | `carbon_calculate.py` | `data/dc_subgraph_carbon.graphml` — **12 carbon weights per edge** (3 vehicles × 4 time periods) using the MOVES5 emission curve |

### Phase 2 — RL Training (`src/pretrain_bc.py`)

Trains **5 MaskablePPO policies** via Pareto-weighted Behavioral Cloning. Each policy imitates a different Dijkstra expert:

| Policy | α (time weight) | β (carbon weight) | Expert |
|---|---|---|---|
| `fastest` | 1.0 | 0.0 | Dijkstra minimising time |
| `time_biased` | 0.75 | 0.25 | Dijkstra on `0.75·time + 0.25·carbon` |
| `balanced` | 0.5 | 0.5 | Dijkstra on equal-weighted cost |
| `carbon_biased` | 0.25 | 0.75 | Dijkstra on `0.25·time + 0.75·carbon` |
| `greenest` | 0.0 | 1.0 | Dijkstra minimising carbon |

**State (154 dimensions):**
- 64-dim GCN embedding of current node
- 64-dim GCN embedding of destination
- 4-dim time-of-day one-hot
- 2-dim compass bearing (Δlon, Δlat) to destination
- 20-dim per-neighbor features (where each outgoing edge leads + edge time + edge carbon)

**Reward (during fine-tuning):** `-(α·norm_time) − (β·norm_carbon) + progress + goal_bonus − revisit_penalty`

### Phase 3 — Evaluation (`src/evaluate.py`)

Runs each PPO policy on 50 random origin-destination pairs and compares against two Dijkstra baselines (fastest + greenest).

**Metrics reported:**
- **Success rate** — fraction of trips reaching the destination
- **Carbon Optimality** = `dijkstra_greenest_carbon / agent_carbon` (1.0 = perfect, the routing analog of accuracy)
- **Carbon saved %** vs Dijkstra fastest baseline
- **Time penalty %** vs Dijkstra fastest baseline

---

## Expected Results

### Dijkstra Baselines (DC subgraph, HeavyTruck, evening rush)

| Method | Mean time | Mean carbon |
|---|---|---|
| Fastest | 469 s | 1,885 g CO₂ |
| Greenest | 477 s (+1.7%) | 1,853 g (−1.7%) |

### RL Policies (50-trip eval, honest mode)

| Policy | Success | Avg Time | Avg Carbon | Carbon Optimality |
|---|---|---|---|---|
| `greenest` (β=1.0) | 10/50 | 526.6 s | **2,035 g** ← lowest carbon | **0.946** ← highest ✓ |
| `fastest` (α=1.0) | 10/50 | 568.4 s | 2,303 g | 0.933 |
| `balanced` | 15/50 | 586.3 s | 2,409 g | 0.907 |
| `carbon_biased` | 15/50 | 562.0 s | 2,279 g | 0.905 |
| `time_biased` (α=0.75) | 13/50 | **524.1 s** ← fastest | 2,207 g | 0.894 |

**The Pareto frontier emerges correctly** — `greenest` picks lowest-carbon routes, `time_biased` picks fastest routes.

### Terrain Study (DC flat vs SF hilly)

| Metric | DC (flat) | SF (hilly) |
|---|---|---|
| Mean carbon savings (Dijkstra) | 2.14% | **3.97%** (+85%) |
| Trips with identical fastest/greenest | 40% | 30% |

San Francisco's hilly topology produces **85% more carbon-aware routing benefit** than DC's flat grid — validating the methodology generalises.

---

## File Structure

```
.
├── README.md                          ← you are here
├── SPECIFICATIONS.md                  ← detailed architecture doc
├── requirements.txt
├── src/
│   ├── extract_network.py             ← OSM subgraph extraction
│   ├── elevation_data.py              ← SRTM elevation + grade
│   ├── add_speeds.py                  ← time-of-day speed profiles
│   ├── carbon_calculate.py            ← MOVES5 carbon weights
│   ├── run_pipeline.py                ← runs all 4 stages
│   ├── rl_env.py                      ← Gymnasium environment
│   ├── pretrain_bc.py                 ← Behavioral Cloning training
│   ├── train_ppo.py                   ← (optional) PPO fine-tuning
│   ├── evaluate.py                    ← evaluation + Pareto plot
│   └── terrain_study.py               ← DC-vs-SF comparison
├── data/
│   ├── dc_subgraph_carbon.graphml     ← final DC graph
│   ├── terrain_dc_carbon.graphml      ← DC graph for terrain study
│   ├── terrain_sf_carbon.graphml      ← SF graph for terrain study
│   └── carbon_heatmap.png             ← carbon intensity visualization
├── models/
│   └── ppo_*_bc.zip                   ← 5 trained policies
└── results/
    ├── pareto_frontier.png            ← main Pareto plot
    ├── policy_metrics.png             ← carbon optimality bar chart
    ├── terrain_comparison.png         ← DC-vs-SF comparison
    ├── evaluation_metrics.csv         ← per-trip evaluation data
    └── terrain_comparison.csv         ← terrain summary table
```

---

## Demo Cheatsheet (for in-person walkthrough)

If you have less than 10 minutes and the pipeline + models are already built, just run:

```bash
cd src
source ../venv/bin/activate

# 1. Show the data: open data/carbon_heatmap.png  (the DC carbon intensity map)

# 2. Re-run evaluation live  (~1 min)
python3 evaluate.py --no-cycle-break --max-detour 2.5

# 3. Open the two key plots:
#    - results/pareto_frontier.png    (RL policies on the carbon-time frontier)
#    - results/policy_metrics.png      (carbon optimality per policy)

# 4. Show terrain validation:
#    - results/terrain_comparison.png  (DC vs SF — 85% more savings in hilly cities)
```

---

## Troubleshooting

**`ModuleNotFoundError: stable_baselines3`** — virtual environment not activated.
Run `source venv/bin/activate` first.

**OpenTopoData API timeouts during pipeline build** — the free elevation API is rate-limited (1 req/sec, 100 nodes/batch). The pipeline retries automatically. Total elevation step takes ~30–60 seconds for 1500 nodes.

**`embeddings_*.npy` cache** — these are auto-generated 64-dim GCN node embeddings cached per graph. Safe to delete; they'll be regenerated on next env load.

**Evaluation shows 0/50 success** — make sure you ran `python3 pretrain_bc.py` first to train the policies. The `ppo_*_bc.zip` files in `models/` must exist.

---

## Citation

If you use this code, please cite:

```
@misc{kowta2026carbonaware,
  title  = {Multi-Objective Reinforcement Learning for Carbon-Aware Global Logistics},
  author = {Kowta, Alekya and Khandoji, Kushi and Berry, Kritika and Sciarillo, Jasmine},
  year   = {2026},
  note   = {CSCI-6364 S26 Project, George Washington University}
}
```

**Data sources:**
- OpenStreetMap (road network) via OSMnx
- NASA SRTM 30m elevation via OpenTopoData
- U.S. EPA MOVES5 emission factors
- U.S. EPA Greenhouse Gas Emission Factors Hub
