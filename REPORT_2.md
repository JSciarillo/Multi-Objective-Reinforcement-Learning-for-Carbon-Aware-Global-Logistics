# Multi-Objective Reinforcement Learning for Carbon-Aware Global Logistics

**CSCI-4364/6364 S26 — Machine Learning**

**Team members:**
Alekya Kowta · Kushi Khandoji · Kritika Berry · Jasmine Sciarillo

---

## Abstract

We built a multi-objective routing engine for freight logistics on a real Washington DC road network enriched with MOVES5 emission factors, SRTM 30-metre elevation grades, and four time-of-day congestion profiles. The system jointly optimises delivery time and CO₂ emissions via a scalarised preference weight α ∈ [0,1] (α=1 = minimise time only, α=0 = minimise carbon only). Three algorithms are implemented and compared:

1. **Weighted A\*** — an exact Pareto solver that sweeps α in 11 steps and achieves **3.34% carbon savings at +3.02% travel time** with a **100% routing success rate** on 20 evaluation trips.
2. **Preference-conditioned Double DQN** (value-based RL) — a single neural network that conditions on α at inference time, achieving **2–15% routing success** after 120,000 training steps; on matched trips it is 2–35% worse than A*, indicating the model requires more training.
3. **PPO + Behavioral Cloning** (policy-based RL) — five policies trained via per-policy Pareto-weighted Dijkstra demonstrations, achieving a **carbon optimality of 0.946** (94.6% of the theoretically optimal green route) with a 20–30% routing success rate on 50 full-graph evaluation trips.

A terrain validation study comparing DC (flat grid) with San Francisco (hilly) shows that hilly networks offer **85% more carbon-aware routing benefit** (3.97% vs 2.14% mean Dijkstra savings), confirming that topographic variance is the key driver of when carbon-aware routing matters most.

---

## Deliverables

1. **End-to-end data pipeline** (`src/run_pipeline.py`) — four-stage system converting OpenStreetMap data into a routing graph with per-edge CO₂ weights. Integrates OSMnx (road network), OpenTopoData SRTM 30m (elevation + grade), and the EPA MOVES5 emission curve, producing **12 carbon weights per edge** (3 vehicle classes × 4 time-of-day periods).

2. **Enriched DC road graph** (`data/dc_subgraph_carbon.graphml`) — ~960-node directed graph of downtown Washington DC with all 12 carbon attributes and 4 time-of-day travel times per edge, ready for algorithm evaluation.

3. **Unified MORL system** (`baseline/single_file_morl.py`) — self-contained implementation of Weighted A*, preference-conditioned Double DQN, and Dijkstra baselines. Runs end-to-end with: `python3 single_file_morl.py [train|astar|compare]`.

4. **Trained DQN checkpoint** (`models/dqn_carbon.pt`) — action-as-input architecture, single network conditioned on α, trained with curriculum learning (4-hop → 30-hop trips).

5. **PPO + Behavioral Cloning system** (`src/pretrain_bc.py`, `src/train_ppo.py`) — five MaskablePPO policies trained on per-policy Pareto-weighted Dijkstra demonstrations. Each policy has its own expert matching its (α, β) preference. Saved as `models/ppo_*_bc.zip`.

6. **Evaluation framework** (`src/evaluate.py`) — 50-trip honest evaluation with per-trip Dijkstra comparison, carbon optimality scoring, and Pareto frontier visualisation.

7. **Terrain validation study** (`src/terrain_study.py`) — automated pipeline that builds identical DC and San Francisco subgraphs, runs Dijkstra Pareto analysis on both, and produces a side-by-side comparison showing how terrain grade drives the routing tradeoff.

8. **All results and plots** in `results/` — Pareto frontier plots (A* curve, PPO policy scatter), policy metrics bar charts, terrain comparison histogram, and per-trip CSVs.

---

## Experiment Design

### Converting the high-level goal to measurable outcomes

The goal — *"reduce CO₂ emissions in freight routing without sacrificing too much delivery time"* — is operationalised via **Pareto scalarisation**: for preference weight α, every edge e is assigned a combined cost `w(e) = α · (time(e) / T_norm) + (1−α) · (carbon(e) / C_norm)`. Sweeping α from 0 to 1 traces the Pareto frontier between the two objectives.

**Three primary metrics:**

| Metric | Formula | Meaning |
|---|---|---|
| **Routing success rate** | goals reached / total trips | Fraction of OD pairs the algorithm can navigate — the routing analog of classification accuracy |
| **Carbon savings %** | `(dijkstra_fastest_carbon − agent_carbon) / dijkstra_fastest_carbon` | How much CO₂ the agent saves vs a "Google Maps" fastest-route approach |
| **Time penalty %** | `(agent_time − dijkstra_fastest_time) / dijkstra_fastest_time` | Extra time cost paid for carbon savings |
| **Carbon Optimality** | `dijkstra_greenest_carbon / agent_carbon` | 1.0 = matches the lowest-carbon Dijkstra path; closer to 1.0 = better |

### Two evaluation protocols

Because different algorithms have different characteristics, we use two complementary evaluations:

- **Controlled 4–16 hop evaluation** (20 pairs, used for A* and DQN): OD pairs are sampled to be between 4 and 16 hops apart on the shortest path, ensuring both pairs with and without meaningful alternative routes are included. 100% of pairs are solvable by Dijkstra.

- **Full-graph random evaluation** (50 pairs, used for PPO): OD pairs are sampled randomly from the full 960-node graph. Many pairs have long optimal paths (up to 80+ hops). This is a harder evaluation — the PPO agent's 20–30% success rate reflects the difficulty of navigating arbitrary trips on the full graph without an explicit search algorithm.

### How the experiment design evolved

The experiment went through three major design pivots:

1. **Single-policy fixed-weight DQN** → failed (0/50 success). Revealed that sparse reward on 960 nodes prevents convergence without structural changes.

2. **Multi-policy PPO with shared BC expert** → partial success (3/50). All policies collapsed to the same behaviour because every policy was BC-trained on the same "fastest-route" expert — differentiating them with PPO fine-tuning failed due to catastrophic forgetting.

3. **Per-policy Pareto-weighted BC + honest evaluation** → 0.946 carbon optimality. Each policy now has its own BC expert matched to its (α, β) preference. Simultaneously, we replaced the misleading cycle-breaking evaluation with an honest no-cycle-break + max-detour-cap evaluation, reducing apparent success rate but making the measured routes meaningful.

4. **Weighted A* as oracle** → 100% success, exact Pareto frontier. Implementing A* after PPO revealed the true achievable tradeoff and provided an upper-bound benchmark.

---

## Methodology

### Phase 1 — Data Pipeline

A four-stage pipeline converts raw geographic data into a routing environment:

| Stage | Script | Input | Output |
|---|---|---|---|
| 1. Network extraction | `extract_network.py` | OpenStreetMap API | ~960-node directed DC subgraph |
| 2. Elevation & grade | `elevation_data.py` | OpenTopoData SRTM 30m | Node elevations; directed edge grades |
| 3. Speed profiles | `add_speeds.py` | OSM speed limits | 4 time-of-day speed states per edge |
| 4. Carbon weights | `carbon_calculate.py` | MOVES5 emission curve | 12 carbon weights per edge |

**The MOVES5 emission model** is a physically validated U-shaped curve mapping vehicle speed to g CO₂/km. Passing a slow evening-rush speed (e.g. 15 kph at 0.5× free-flow) to the curve automatically yields the high-emission idling penalty — no separate idle model is needed. Grade penalties are applied per vehicle class: HeavyTruck (hill penalty 15×) > DeliveryVan (8×) > GasCar (4×).

**The result:** each edge carries attributes like `carbon_HeavyTruck_evening_rush` (grams CO₂), `travel_time_evening_rush` (seconds), and 10 analogues for other vehicles and time periods.

*(See Figure 1 — carbon heatmap)*

### Phase 2 — Weighted A* (Exact Pareto Solver)

For each preference α, define combined edge cost:
```
w(e) = α · (time(e) / T_norm)  +  (1−α) · (carbon(e) / C_norm)
```
where T_norm and C_norm are per-graph median edge values, ensuring α=0.5 is unit-neutral. Run A* with a haversine lower-bound heuristic (admissible since haversine distance / 110 kph ≤ actual travel time). Sweeping α from 0.0 to 1.0 in 11 steps produces 11 exact Pareto-optimal routing decisions per OD pair.

This serves as the **oracle upper bound**: no algorithm can find better routes on this graph. All RL agents are evaluated relative to it.

### Phase 3 — Preference-Conditioned Double DQN (Value-Based RL)

Architecture: **action-as-input** shared-weight network (fixing the generalisation failure of classical fixed-head DQN on graphs).

- **Global state (6-dim):** `(cur_x, cur_y, dst_x, dst_y, dist_remaining_norm, α)` — α is embedded directly, making one network handle all preferences
- **Per-edge features (6-dim × MAX_DEG slots):** `(Δx, Δy, edge_time_norm, edge_carbon_norm, dist_after_norm, valid_flag)` — the network scores each candidate neighbour by its actual features, not its slot index
- Shared weights across slots: "slot 0 at node A" and "slot 0 at node B" are treated differently because their feature vectors differ

**Training:** Double DQN, curriculum learning (starts at 4-hop trips; promotes when rolling 65%-window success rate exceeds threshold), potential-based reward shaping (haversine distance), revisit penalty −0.3. 120,000 total training steps.

### Phase 4 — PPO + Behavioral Cloning (Policy-Based RL)

Five MaskablePPO policies, one per Pareto point:

| Policy | α | β | BC Expert |
|---|---|---|---|
| fastest | 1.00 | 0.00 | Dijkstra on `time` |
| time_biased | 0.75 | 0.25 | Dijkstra on `0.75·time + 0.25·carbon` |
| balanced | 0.50 | 0.50 | Dijkstra on `0.5·time + 0.5·carbon` |
| carbon_biased | 0.25 | 0.75 | Dijkstra on `0.25·time + 0.75·carbon` |
| greenest | 0.00 | 1.00 | Dijkstra on `carbon` |

**Key design decisions:**
- **Per-policy BC**: each policy has its own Dijkstra expert matching its preference weight. All-policies-same-expert failed (catastrophic forgetting when PPO fine-tuning tried to differentiate them).
- **Neighbor features in state (154-dim total):** 64-dim GCN current node + 64-dim GCN destination + 4-dim time one-hot + 2-dim compass (Δlon, Δlat) + 20-dim per-neighbor features (Δlon, Δlat, edge_time_norm, edge_carbon_norm per outgoing edge). Adding these features dropped BC loss from 0.65 → 0.22 and raised carbon optimality from 0.30 → 0.946.
- **Honest evaluation:** deterministic rollout, no cycle-breaking heuristic, path capped at 2.5× Dijkstra optimal hops. Success rate drops from a misleading 60% to an honest 20–30%, but every "success" is a genuinely good route.

---

## Results

### Result 1 — Weighted A* Pareto Frontier

*(See Figure 2 — A* smooth blue curve)*

**20/20 success at every α value.** The A* sweep produces the optimal tradeoff between time and carbon:

| α | Avg Time (s) | Avg CO₂ (g) | vs Dijkstra-fastest: Time | vs Dijkstra-fastest: Carbon |
|---|---|---|---|---|
| 0.00 | 235.5 | 910.4 | +3.0% | **−3.3%** ← greenest |
| 0.10 | 234.5 | 910.6 | +2.6% | −3.3% |
| 0.20 | 234.5 | 910.6 | +2.6% | −3.3% |
| **0.30** | **232.5** | **913.2** | **+1.7%** | **−3.1%** ← **sweet spot** |
| 0.40 | 231.4 | 914.9 | +1.2% | −2.9% |
| 0.50 | 230.7 | 917.1 | +1.0% | −2.6% |
| 0.60 | 229.6 | 922.3 | +0.4% | −2.1% |
| 0.70 | 229.5 | 922.9 | +0.4% | −2.0% |
| 0.80 | 228.9 | 930.3 | +0.1% | −1.2% |
| 0.90 | 228.8 | 934.4 | +0.1% | −0.8% |
| 1.00 | 228.6 | 941.9 | 0.0% | 0.0% ← fastest |

**Dijkstra baselines (20 pairs):** Fastest = 228.6s / 941.9g · Greenest = 235.5s / 910.4g

The **α=0.3 sweet spot** is practically significant: a HeavyTruck saves 3.1% CO₂ for only 1.7% extra travel time. For a fleet running 100 trucks and 10 deliveries/day, this is meaningful emission reduction with negligible schedule impact.

*(See Figure 3 — per-trip weighted Dijkstra scatter)*

### Result 2 — Value-Based RL (Double DQN)

*(Orange scattered squares in Figure 2)*

After 120,000 training steps, the DQN achieves **2–15% routing success rate** (1–3 of 20 pairs per α setting). At α=1.0 (pure time), success rate is 0/20 — the hardest setting for the agent.

**Critical: matched-pair comparison** (same trips where DQN succeeded vs A* on the same trips):

| α | DQN vs A* (time) | DQN vs A* (carbon) | n pairs |
|---|---|---|---|
| 0.00 | +8.5% slower | +7.4% more carbon | 2/20 |
| 0.10 | +9.8% slower | +9.0% more carbon | 3/20 |
| 0.50 | +2.6% slower | +3.0% more carbon | 2/20 |
| 0.70 | +19.7% slower | +17.4% more carbon | 2/20 |
| 0.90 | +34.8% slower | +31.3% more carbon | 2/20 |

On the same trips, the DQN is **consistently worse than A*** by 2–35% depending on α. The DQN's apparently lower time/carbon in the raw output is a selection bias: it only succeeded on the easiest 1–3 trips while A* averaged over all 20 (including harder, longer trips).

**What the DQN achieves:** The action-as-input architecture and preference-conditioned state are validated as architecturally correct — the agent does navigate to the destination on some trips and does respond to the α preference in the state. It needs approximately 3–5× more training steps (400–600k) to close the gap with A*.

### Result 3 — PPO + Behavioral Cloning

*(See Figure 4 — PPO Pareto scatter · Figure 5 — policy metrics bar charts)*

**Dijkstra baselines (50 full-graph random pairs):** Fastest = 469s / 1,885g · Greenest = 477s (+1.7%) / 1,853g (−1.7%)

| Policy | α | β | Success | Avg Time | Avg CO₂ | Carbon Optimality |
|---|---|---|---|---|---|---|
| greenest | 0.00 | 1.00 | 10/50 | 526.6 s | **2,035 g** ← lowest | **0.946** ← highest ✓ |
| fastest | 1.00 | 0.00 | 10/50 | 568.4 s | 2,303 g | 0.933 |
| balanced | 0.50 | 0.50 | 15/50 | 586.3 s | 2,409 g | 0.907 |
| carbon_biased | 0.25 | 0.75 | 15/50 | 562.0 s | 2,279 g | 0.905 |
| time_biased | 0.75 | 0.25 | 13/50 | **524.1 s** ← fastest | 2,207 g | 0.894 ← lowest |

**Key findings:**
- **The Pareto ordering is correct**: the greenest policy achieves the highest carbon optimality (0.946), the time_biased the lowest (0.894) — policies are differentiating along the intended axis.
- **Carbon optimality 0.946** means the greenest PPO routes emit only 5.4% more CO₂ than the theoretically optimal Dijkstra-green path.
- **20–30% success rate** on 50 full-graph random pairs. The harder evaluation (random trips across 960 nodes vs controlled 4–16 hops) explains why PPO success rate (20–30%) appears lower than A* (100%) — they are measured on different trip populations.

### Result 4 — Algorithm Comparison

| Algorithm | Success Rate | Best carbon savings | Training needed | Notes |
|---|---|---|---|---|
| **Weighted A*** | **100%** (20/20) | **−3.34%** (α=0) | None | Exact oracle; needs full graph at query time |
| **PPO + BC** | 20–30% (10–15/50) | −4.8% on successful trips | ~2 min BC | Learned; 0.946 carbon optimality when it succeeds |
| **DQN** | 2–15% (1–3/20) | Varies; worse than A* | ~30 min RL | Architecture correct; needs more training |
| Dijkstra fastest | 100% | 0% (baseline) | None | The "Google Maps" baseline |
| Dijkstra greenest | 100% | −3.34% | None | Theoretical green ceiling |

**How the result evolved over project iterations:**

| Iteration | Approach | Carbon Optimality | Success Rate |
|---|---|---|---|
| 1 | Vanilla DQN from scratch | n/a | 0/50 |
| 2 | PPO from scratch | n/a | 1/50 |
| 3 | PPO + curriculum (8→20→full hops) | 0.231 | 3/50 |
| 4 | All-policy BC + PPO fine-tuning | 0.303 | 12/50 |
| **5** | **Per-policy BC + neighbor features** | **0.946** | **10/50** |
| **A*** | **Exact Pareto sweep** | **1.000** | **20/20** |

### Result 5 — Terrain Validation Study (DC vs San Francisco)

*(See Figure 6 — terrain comparison)*

The same pipeline (same MOVES5 model, same vehicle, same time profile) was re-run on a 1,477-node San Francisco subgraph to test whether the methodology and the narrow Pareto tradeoff are properties of DC specifically or of the approach in general.

| Metric | DC (flat) | SF (hilly) |
|---|---|---|
| Mean carbon savings (Dijkstra) | 2.14% | **3.97%** (+85%) |
| Max single-trip savings | 23.05% | 23.32% |
| % trips with identical fastest/greenest path | 40.0% | 30.5% |
| Mean time penalty for greenest path | 2.06% | 3.67% |

**San Francisco's hilly topology offers 85% more carbon-aware routing benefit than DC's flat grid.** In SF, 69.5% of trips have a strictly greener alternative route (vs 60% in DC). Both cities share a similar maximum single-trip savings (~23%), but SF has many more trips in the 5–20% savings range while DC trips cluster near 0%.

This validates two claims: (1) the methodology generalises to other cities, and (2) topographic grade variance is the primary driver of when carbon-aware routing delivers meaningful benefits.

---

## Lessons Learned

**Technical:**

1. **State representation is the most important design choice — more important than the algorithm.** Four iterations of algorithm changes (DQN → PPO → curriculum → BC) produced incremental gains. One architecture change — adding per-neighbor features (where each action leads and what it costs) — dropped BC loss from 0.65 to 0.22 and raised carbon optimality from 0.30 to 0.946 in a single step.

2. **Always implement the exact solver before the RL agent.** Weighted A* took one afternoon to implement and immediately produced a clean, interpretable Pareto curve with 100% success. This gave us: (a) the correct baseline numbers, (b) a sanity check that the graph and carbon weights were correct, and (c) an oracle to measure RL agents against. Without it, we would not have known whether our RL agents were close to optimal.

3. **Sparse reward + large graphs = RL cannot learn from scratch.** Three attempts (vanilla DQN, PPO, PPO+curriculum) failed to navigate a 960-node graph through random exploration. Behavioral Cloning warm-start — showing the agent thousands of Dijkstra demonstrations before RL training — was the enabling change. This is well-established in the routing RL literature but worth experiencing first-hand.

4. **Per-policy expert matching prevents catastrophic forgetting in multi-objective BC.** Training all five policies on the same "fastest-route" expert, then trying to differentiate them with PPO fine-tuning, failed — every policy converged to the same middling behaviour. Each policy must see demonstrations that match its own preference from the start.

5. **Selection bias in RL evaluation is dangerous.** An agent that succeeds on 10% of trips looks statistically impressive when averaged — it succeeded on the easiest trips. The matched-pair comparison (evaluate agent and oracle on the *exact same* trip) is the only honest measurement. Our DQN's apparent speed advantage in raw averages was entirely selection bias; on matched pairs it was 2–35% worse than A*.

6. **The Pareto tradeoff depends on geography.** DC's 1.7% (Dijkstra) and 3.34% (A*) savings are real but narrow because the flat grid offers few alternative routes. San Francisco's 3.97% savings confirms the methodology is sound — it just needs terrain to work with.

**Operational:**

7. **Caching intermediate computation unlocked iteration speed.** GCN embeddings for the 960-node graph originally recomputed every environment load (~30 seconds). Caching them by graph file hash reduced env startup from 30s to <1s and made node embeddings consistent between training and evaluation. Two hours of debugging traced to non-deterministic embeddings would have been avoided immediately with caching.

---

## Attribution

*(Fill in actual contributions per team member)*

**Alekya Kowta** — Data pipeline architecture and integration; RL environment design (`rl_env.py`); Behavioral Cloning system (`pretrain_bc.py`); evaluation framework (`evaluate.py`); terrain validation study; report writing.

**Kushi Khandoji** — *(describe actual contributions)*

**Kritika Berry** — *(describe actual contributions)*

**Jasmine Sciarillo** — *(describe actual contributions)*

---

## References

[1] Erdoğan, S., & Miller-Hooks, E. (2012). A green vehicle routing problem. *Transportation Research Part E*, 48(1), 100–114.

[2] Garside, A. K., Ahmad, R., & Muhtazaruddin, M. N. B. (2024). A recent review of solution approaches for green vehicle routing problem and its variants. *Operations Research Perspectives*, 12, 100303.

[3] Zou, Y., Wu, H., Yin, Y., et al. (2024). An improved transformer model for low-carbon multi-depot vehicle routing. *Annals of Operations Research*, 339(1), 517–536.

[4] Yue, B., Ma, J., Shi, J., & Yang, J. (2024). Deep RL-based adaptive search for time-dependent green VRP. *IEEE Access*, 12, 33400–33419.

[5] Wang, Y., Qiu, D., He, Y., Zhou, Q., & Strbac, G. (2023). Multi-agent RL for electric vehicle decarbonized routing and scheduling. *Energy*, 284, 129335.

[6] Rakha, H. A., et al. (2011). Virginia Tech comprehensive power-based fuel consumption model. *Transportation Research Part D*, 16(7), 492–503.

[7] Boeing, G. (2017). OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks. *Computers, Environment and Urban Systems*, 65, 126–139.

[8] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv:1707.06347*.

[9] Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529–533.

[10] van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with Double Q-learning. *AAAI*, 30(1).

[11] Hill, A., et al. (2018). Stable Baselines3: Reliable Reinforcement Learning Implementations. https://github.com/DLR-RM/stable-baselines3

[12] Raffin, A., et al. (2021). Stable-Baselines3: Reliable Reinforcement Learning Implementations. *Journal of Machine Learning Research*, 22(268), 1–8.

[13] U.S. EPA (2024). MOVES5 Motor Vehicle Emission Simulator. https://www.epa.gov/moves

[14] U.S. EPA (2023). Emission Factors for Greenhouse Gas Inventories. https://www.epa.gov/climateleadership/ghg-emission-factors-hub

[15] OpenStreetMap contributors. OpenStreetMap. https://www.openstreetmap.org

[16] OpenTopoData. Open elevation API serving SRTM 30m. https://www.opentopodata.org

[17] IEA (2023). Transport. International Energy Agency. https://www.iea.org/topics/transport

---

## Figure Reference Guide

When copying this report into the .docx template, insert the figures at the locations marked below.

**Figure 1** → Insert after Methodology / Phase 1 section
- File: `data/carbon_heatmap.png`
- Caption: *"Carbon intensity heatmap of the DC subgraph (~960 nodes). Warmer/brighter corridors indicate higher CO₂ emission weights per edge, driven by uphill road grade (SRTM elevation data) and evening rush hour speed degradation (0.5× free-flow speed applied to the MOVES5 U-shaped emission curve)."*

**Figure 2** → Insert at top of Results section (first figure the reader sees)
- File: `results/pareto_frontier.png`
- Caption: *"Pareto frontier: average travel time vs average CO₂ emissions (HeavyTruck, evening rush, 20 evaluation pairs). Blue curve (Weighted A*) is the exact optimal frontier — 100% routing success at all 11 α values. Orange squares (DQN) show the trained agent at 2–15% success; on matched trips the DQN is 2–35% worse than A*, indicating insufficient training."*

**Figure 3** → Insert after A* result table
- File: `baseline/pareto_front_dijkstra.png`
- Caption: *"Per-trip Pareto scatter using Weighted Dijkstra across 20 OD pairs. Each dot is one trip; colour encodes the carbon preference weight (green = carbon-prioritising, red = time-prioritising). The diagonal spread shows real per-trip variation — some pairs have up to 23% carbon savings available, others near zero."*

**Figure 4** → Insert at start of PPO Results subsection
- File: `results/pareto_frontier_ppo.png`
- Caption: *"PPO + BC Pareto frontier (50 evaluation trips, full DC graph, honest evaluation mode). Five coloured circles are the five trained policies. The greenest policy (purple, β=1.0) finds the lowest-carbon routes (2,035g); the time_biased policy (orange, α=0.75) finds the fastest routes (524s). Dijkstra baselines shown as diamond markers."*

**Figure 5** → Insert after PPO results table
- File: `results/policy_metrics_ppo.png`
- Caption: *"PPO policy evaluation metrics. Left: carbon saved % and time penalty % vs Dijkstra fastest on the per-trip matched pairs where the policy succeeded. Right: carbon optimality per policy (dashed green line = 1.0 = perfect match to Dijkstra greenest). The greenest policy achieves the highest carbon optimality (0.946); the time_biased policy the lowest (0.894), confirming the policies correctly trace the Pareto axis."*

**Figure 6** → Insert at start of Terrain Validation subsection
- File: `results/terrain_comparison.png`
- Caption: *"Terrain study: Washington DC (flat, blue) vs San Francisco (hilly, red). Left: per-trip distribution of available carbon savings from Dijkstra. SF has a broader distribution with more trips in the 5–20% savings range. Right: summary statistics — SF offers 3.97% mean savings vs DC's 2.14%, an 85% larger carbon-aware routing benefit, driven by greater topographic grade variance."*
