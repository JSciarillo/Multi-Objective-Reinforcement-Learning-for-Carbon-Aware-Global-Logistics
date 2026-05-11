"""
Evaluates trained PPO Pareto policies against Dijkstra baselines.

Produces:
  - results/evaluation_metrics.csv   — per-trip numbers for every policy
  - results/pareto_frontier.png      — time vs carbon scatter (the Pareto plot)

Usage:
    cd src
    python evaluate.py                  # all 5 policies, 50 trips each
    python evaluate.py --trips 20       # faster smoke-test
"""

import argparse
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from rl_env import CarbonRoutingEnv

GRAPH_PATH  = str(Path(__file__).parent.parent / "data" / "dc_subgraph_carbon.graphml")
MODELS_DIR  = Path(__file__).parent.parent / "models"
RESULTS_DIR = Path(__file__).parent.parent / "results"

PARETO_LABELS = ["fastest", "time_biased", "balanced", "carbon_biased", "greenest"]
PARETO_ALPHAS = {
    "fastest":       (1.00, 0.00),
    "time_biased":   (0.75, 0.25),
    "balanced":      (0.50, 0.50),
    "carbon_biased": (0.25, 0.75),
    "greenest":      (0.00, 1.00),
}

# -----------------------------------------------------------------------
# Baseline planners
# -----------------------------------------------------------------------

def _ensure_float(G, key, default=10.0):
    for _, _, _, d in G.edges(keys=True, data=True):
        raw = d.get(key)
        if raw is None:
            d[key] = default
        else:
            try:
                d[key] = float(raw)
            except (ValueError, TypeError):
                d[key] = default


def dijkstra_path(G, src, dst, weight_key):
    _ensure_float(G, weight_key)
    try:
        return nx.shortest_path(G, src, dst, weight=weight_key)
    except nx.NetworkXNoPath:
        return None


def path_metrics(G, path, vehicle, time_profile):
    time_key   = f"travel_time_{time_profile}"
    carbon_key = f"carbon_{vehicle}_{time_profile}"
    total_t = total_c = 0.0
    for u, v in zip(path[:-1], path[1:]):
        edge = G.get_edge_data(u, v)
        if edge is None:
            continue
        d = edge[min(edge)]
        t = float(d.get(time_key, 10.0))
        total_t += t if t != float("inf") else 1000.0
        total_c += float(d.get(carbon_key, 10.0))
    return total_t, total_c


# -----------------------------------------------------------------------
# PPO runner
# -----------------------------------------------------------------------

def run_ppo_episode(model, env, start, dest, cycle_break=True, max_detour=None):
    """
    Run a single PPO episode.
      cycle_break=True  : if greedy action would revisit, substitute unvisited neighbour
      max_detour=K      : abort as failure if path exceeds K * dijkstra_hops
    """
    obs, _ = env.reset(options={"start": start, "dest": dest})

    # Optional detour cap based on Dijkstra optimal hop count
    detour_limit = None
    if max_detour is not None:
        try:
            optimal_hops = nx.shortest_path_length(env.G, start, dest)
            detour_limit = int(max_detour * optimal_hops) + 1
        except nx.NetworkXNoPath:
            detour_limit = None

    path = [env.current_node]
    visited = {env.current_node}
    done = False

    while not done:
        masks = get_action_masks(env)
        action, _ = model.predict(obs, action_masks=masks, deterministic=True)
        action = int(action)

        if cycle_break:
            # If greedy action would revisit a node, pick an unvisited neighbour instead.
            out_edges = list(env.G.out_edges(env.current_node, keys=True))
            if action < len(out_edges):
                next_node = out_edges[action][1]
                if next_node in visited:
                    for alt in range(len(out_edges)):
                        if out_edges[alt][1] not in visited:
                            action = alt
                            break

        obs, _, terminated, truncated, _ = env.step(action)
        visited.add(env.current_node)
        path.append(env.current_node)
        done = terminated or truncated

        if detour_limit is not None and len(path) > detour_limit:
            return path, False  # detour cap exceeded → failure

    success = env.current_node == dest
    return path, success


# -----------------------------------------------------------------------
# Main evaluation
# -----------------------------------------------------------------------

def run_evaluation(
    graph_path=GRAPH_PATH,
    num_trips=50,
    vehicle="HeavyTruck",
    time_profile="evening_rush",
    max_eval_hops=None,   # None = full graph; int = only sample short OD pairs
    cycle_break=True,
    max_detour=None,      # None = no cap; float = cap path at K * dijkstra_hops
):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Build a reference env (shares the graph & embeddings with all policy envs)
    print("Building reference environment...")
    ref_env = CarbonRoutingEnv(
        graph_path=graph_path,
        alpha=0.5,
        beta=0.5,
        vehicle=vehicle,
        time_profile=time_profile,
        max_steps=500,
    )
    G = ref_env.G
    nodes = list(G.nodes())

    # Discover which models are available.
    # Prefer per-policy BC models (principled Pareto-weighted experts) over PPO fine-tuned ones
    # since PPO fine-tuning has been observed to overwrite BC's good routing prior.
    available = {}
    for label in PARETO_LABELS:
        for suffix in ["_bc.zip", ".zip"]:
            p = MODELS_DIR / f"ppo_{label}{suffix}"
            if p.exists():
                available[label] = p
                break
        if label not in available:
            print(f"  [skip] {label}: no model found")

    if not available:
        print("\nNo PPO models found. Run train_ppo.py first.")
        return

    # Load models into per-label envs
    ppo_models = {}
    ppo_envs   = {}
    for label, model_path in available.items():
        alpha, beta = PARETO_ALPHAS[label]
        env = CarbonRoutingEnv(
            graph_path=graph_path,
            alpha=alpha,
            beta=beta,
            vehicle=vehicle,
            time_profile=time_profile,
            max_steps=500,
        )
        # Share pre-computed embeddings to avoid re-training GCN
        env.embeddings = ref_env.embeddings
        ppo_envs[label]   = env
        ppo_models[label] = MaskablePPO.load(str(model_path), env=env)
        print(f"  Loaded {label} from {model_path.name}")

    # Sample valid OD pairs
    hop_note = f" (≤{max_eval_hops} hops)" if max_eval_hops else " (full graph)"
    print(f"\nSampling {num_trips} valid OD pairs{hop_note}...")
    np.random.seed(42)
    random.seed(42)
    time_key   = f"travel_time_{time_profile}"
    carbon_key = f"carbon_{vehicle}_{time_profile}"

    od_pairs = []
    attempts = 0
    while len(od_pairs) < num_trips and attempts < num_trips * 30:
        attempts += 1
        src = random.choice(nodes)
        dst = random.choice(nodes)
        if src == dst:
            continue
        if max_eval_hops is not None:
            try:
                hops = nx.shortest_path_length(G, src, dst)
                if hops > max_eval_hops:
                    continue
            except nx.NetworkXNoPath:
                continue
        p = dijkstra_path(G, src, dst, time_key)
        if p is not None:
            od_pairs.append((src, dst))

    print(f"Found {len(od_pairs)} valid pairs in {attempts} attempts.")

    # Evaluate
    rows = []
    for trip_id, (src, dst) in enumerate(od_pairs, 1):
        row = {"trip_id": trip_id, "start": src, "dest": dst}

        # Dijkstra baselines
        fast_path = dijkstra_path(G, src, dst, time_key)
        green_path = dijkstra_path(G, src, dst, carbon_key)

        ft, fc = path_metrics(G, fast_path,  vehicle, time_profile) if fast_path  else (None, None)
        gt, gc = path_metrics(G, green_path, vehicle, time_profile) if green_path else (None, None)

        row["dijkstra_fastest_time_s"]   = ft
        row["dijkstra_fastest_carbon_g"] = fc
        row["dijkstra_greenest_time_s"]  = gt
        row["dijkstra_greenest_carbon_g"]= gc

        # PPO policies
        for label, model in ppo_models.items():
            env = ppo_envs[label]
            path, success = run_ppo_episode(model, env, src, dst,
                                              cycle_break=cycle_break, max_detour=max_detour)
            if success:
                t, c = path_metrics(G, path, vehicle, time_profile)
                row[f"ppo_{label}_time_s"]   = t
                row[f"ppo_{label}_carbon_g"] = c
                row[f"ppo_{label}_success"]  = True
            else:
                row[f"ppo_{label}_time_s"]   = None
                row[f"ppo_{label}_carbon_g"] = None
                row[f"ppo_{label}_success"]  = False

        rows.append(row)

        if trip_id % 10 == 0:
            print(f"  {trip_id}/{len(od_pairs)} trips done")

    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "evaluation_metrics.csv"
    df.to_csv(str(csv_path), index=False)
    print(f"\nMetrics saved → {csv_path}")

    # -----------------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BASELINE COMPARISON")
    print("=" * 60)
    valid = df.dropna(subset=["dijkstra_fastest_time_s"])
    fast_t_avg  = valid["dijkstra_fastest_time_s"].mean()
    fast_c_avg  = valid["dijkstra_fastest_carbon_g"].mean()
    green_t_avg = valid["dijkstra_greenest_time_s"].mean()
    green_c_avg = valid["dijkstra_greenest_carbon_g"].mean()

    carbon_saved_dijk  = fast_c_avg - green_c_avg
    carbon_saved_pct   = carbon_saved_dijk / fast_c_avg * 100
    time_penalty_dijk  = green_t_avg - fast_t_avg
    time_penalty_pct   = time_penalty_dijk / fast_t_avg * 100

    print(f"  Dijkstra Fastest  — time: {fast_t_avg:.1f}s   carbon: {fast_c_avg:.1f}g")
    print(f"  Dijkstra Greenest — time: {green_t_avg:.1f}s   carbon: {green_c_avg:.1f}g")
    print(f"  Carbon saved by going green : {carbon_saved_dijk:.1f}g  ({carbon_saved_pct:.1f}%)")
    print(f"  Time cost  of going green   : +{time_penalty_dijk:.1f}s ({time_penalty_pct:.1f}%)")

    print("\nPPO POLICIES  (per-trip comparison against same trip's Dijkstra result)")
    print("-" * 60)
    print(f"  {'Policy':<15} {'Success':>8}  {'AvgTime':>9}  {'AvgCarbon':>11}  "
          f"{'CarbonSaved%':>13}  {'TimePenalty%':>13}  {'CarbonOptim':>12}")
    ppo_summary = []
    for label in available:
        sdf = df[df[f"ppo_{label}_success"] == True].copy()
        n   = len(sdf)
        if n > 0:
            avg_t = sdf[f"ppo_{label}_time_s"].mean()
            avg_c = sdf[f"ppo_{label}_carbon_g"].mean()

            # Per-trip: compare agent result against THAT trip's Dijkstra result
            per_trip_c_saved  = ((sdf["dijkstra_fastest_carbon_g"] - sdf[f"ppo_{label}_carbon_g"])
                                  / sdf["dijkstra_fastest_carbon_g"] * 100)
            per_trip_t_pen    = ((sdf[f"ppo_{label}_time_s"] - sdf["dijkstra_fastest_time_s"])
                                  / sdf["dijkstra_fastest_time_s"] * 100)
            per_trip_c_opt    = (sdf["dijkstra_greenest_carbon_g"] / sdf[f"ppo_{label}_carbon_g"])

            c_saved_pct  = per_trip_c_saved.mean()
            t_penalty_pct = per_trip_t_pen.mean()
            c_optimality = per_trip_c_opt.mean()
            t_optimality = (sdf["dijkstra_fastest_time_s"] / sdf[f"ppo_{label}_time_s"]).mean()

            ppo_summary.append((label, avg_t, avg_c, n,
                                 c_saved_pct, t_penalty_pct, c_optimality, t_optimality))
            print(f"  {label:<15} {n:>3}/{len(od_pairs):<4}   "
                  f"{avg_t:>8.1f}s  {avg_c:>10.1f}g  "
                  f"{c_saved_pct:>+12.1f}%  {t_penalty_pct:>+12.1f}%  "
                  f"{c_optimality:>11.3f}")
        else:
            print(f"  {label:<15} {0:>3}/{len(od_pairs):<4}   (no successful trips)")

    # -----------------------------------------------------------------------
    # Save enriched metrics and plots
    # -----------------------------------------------------------------------
    _plot_pareto(df, valid, ppo_summary, available, fast_t_avg, fast_c_avg, green_t_avg, green_c_avg)
    if ppo_summary:
        _plot_metrics_bar(ppo_summary, fast_c_avg, green_c_avg)


def _plot_pareto(df, valid, ppo_summary, available,
                 fast_t_avg, fast_c_avg, green_t_avg, green_c_avg):
    fig, ax = plt.subplots(figsize=(9, 6))

    # Dijkstra baselines
    ax.scatter(fast_t_avg,  fast_c_avg,  marker="D", s=140, color="red",   zorder=5, label="Dijkstra: Fastest")
    ax.scatter(green_t_avg, green_c_avg, marker="D", s=140, color="green", zorder=5, label="Dijkstra: Greenest")

    # PPO Pareto points
    colors = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8", "#984ea3"]
    label_order = ["fastest", "time_biased", "balanced", "carbon_biased", "greenest"]
    for (label, avg_t, avg_c, n, c_saved_pct, t_penalty_pct, c_opt, t_opt), color in zip(
        sorted(ppo_summary, key=lambda x: label_order.index(x[0])),
        colors,
    ):
        alpha_val, beta_val = PARETO_ALPHAS[label]
        ax.scatter(avg_t, avg_c, s=180, color=color, zorder=6,
                   label=f"PPO: {label} (α={alpha_val:.2f}, β={beta_val:.2f})  "
                         f"C-saved={c_saved_pct:+.1f}%  T-pen={t_penalty_pct:+.1f}%  n={n}")

    # Connect PPO points to show frontier
    if len(ppo_summary) > 1:
        pts = sorted(ppo_summary, key=lambda x: x[1])  # sort by time
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        ax.plot(xs, ys, "k--", linewidth=1.2, alpha=0.5, zorder=4, label="Pareto frontier (PPO)")

    ax.set_xlabel("Average Travel Time (s)", fontsize=12)
    ax.set_ylabel("Average Carbon Emissions (g CO₂)", fontsize=12)
    ax.set_title("Pareto Frontier: Delivery Time vs Carbon Emissions\n(DC Subgraph, HeavyTruck, Evening Rush)", fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    plot_path = RESULTS_DIR / "pareto_frontier.png"
    fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPareto plot saved → {plot_path}")


def _plot_metrics_bar(ppo_summary, fast_c_avg, green_c_avg):
    """Bar chart: carbon savings % and time penalty % per policy."""
    label_order = ["fastest", "time_biased", "balanced", "carbon_biased", "greenest"]
    ordered = sorted(ppo_summary, key=lambda x: label_order.index(x[0]))

    labels       = [r[0].replace("_", "\n") for r in ordered]
    c_saved_pcts = [r[4] for r in ordered]   # carbon saved vs fastest baseline
    t_penalty_pcts = [r[5] for r in ordered] # time penalty vs fastest baseline
    c_optims     = [r[6] for r in ordered]   # carbon optimality (green Dijkstra / agent)

    x = np.arange(len(labels))
    width = 0.3

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left: carbon saved % and time penalty % ---
    ax = axes[0]
    bars1 = ax.bar(x - width/2, c_saved_pcts,  width, color="#2ca02c", alpha=0.8, label="Carbon saved vs fastest (%)")
    bars2 = ax.bar(x + width/2, t_penalty_pcts, width, color="#d62728", alpha=0.8, label="Time penalty vs fastest (%)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Carbon Savings vs Time Penalty per Policy")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width()/2, h + (0.2 if h >= 0 else -0.2),
                f"{h:+.1f}%", ha="center", va=va, fontsize=8)

    # --- Right: carbon optimality (how close to Dijkstra greenest) ---
    ax2 = axes[1]
    colors = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8", "#984ea3"][:len(ordered)]
    bars = ax2.bar(x, c_optims, color=colors, alpha=0.85)
    ax2.axhline(1.0, color="green", linewidth=1.5, linestyle="--", label="Dijkstra Greenest (1.0 = perfect)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Carbon Optimality  (Dijkstra-green carbon / Agent carbon)")
    ax2.set_title("Carbon Optimality per Policy\n(1.0 = matches the greenest possible Dijkstra route)")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0, max(max(c_optims, default=1), 1.0) * 1.15)
    for bar, val in zip(bars, c_optims):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("PPO Policy Evaluation Metrics  (DC Subgraph, HeavyTruck, Evening Rush)", fontsize=12)
    plt.tight_layout()
    plot_path = RESULTS_DIR / "policy_metrics.png"
    fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Metrics bar chart saved → {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph",    default=GRAPH_PATH)
    parser.add_argument("--trips",    type=int, default=50)
    parser.add_argument("--vehicle",  default="HeavyTruck")
    parser.add_argument("--time",     default="evening_rush")
    parser.add_argument("--max-hops",    type=int,  default=None,
                        help="Only evaluate on OD pairs within N hops (e.g. 15)")
    parser.add_argument("--no-cycle-break", action="store_true",
                        help="Disable evaluation-time cycle breaking (honest policy eval)")
    parser.add_argument("--max-detour",  type=float, default=None,
                        help="Abort as failure if path > K x Dijkstra hops (e.g. 2.5)")
    args = parser.parse_args()

    run_evaluation(
        graph_path=args.graph,
        num_trips=args.trips,
        vehicle=args.vehicle,
        time_profile=args.time,
        max_eval_hops=args.max_hops,
        cycle_break=not args.no_cycle_break,
        max_detour=args.max_detour,
    )
