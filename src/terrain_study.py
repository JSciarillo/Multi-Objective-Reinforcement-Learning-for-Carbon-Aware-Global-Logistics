"""
Terrain comparison study: hilly vs flat carbon-time Pareto room.

Builds equivalent road-network subgraphs for two cities (default: Washington DC
and San Francisco), runs the full carbon-weight pipeline on both, then compares
the theoretical Pareto envelope by running Dijkstra fastest vs greenest on
hundreds of random origin-destination pairs in each.

The claim being tested:
    On flat dense networks (DC), the fastest route is nearly always the greenest
    route, leaving little room for carbon-aware optimization. On hilly networks
    (SF), real Pareto tradeoffs exist because grade and speed-class variance is
    much higher.

Usage:
    cd src
    python terrain_study.py
    # outputs: results/terrain_comparison.png  and  results/terrain_comparison.csv
"""

import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

from extract_network    import extract_subgraph
from elevation_data     import add_elevation_to_graph
from add_speeds         import add_speeds_and_travel_times
from carbon_calculate   import calculate_carbon_weights

DATA_DIR    = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

CITIES = {
    "dc": {
        "name":   "Washington DC",
        "label":  "DC (flat)",
        "center": (38.8977,  -77.0365),
        "color":  "#1f77b4",
    },
    "sf": {
        "name":   "San Francisco",
        "label":  "SF (hilly)",
        "center": (37.7749, -122.4194),
        "color":  "#d62728",
    },
}

# Shared graph parameters so the comparison is apples-to-apples
TARGET_NODES = 1500
RADIUS_M     = 4000
NUM_TRIPS    = 200
VEHICLE      = "HeavyTruck"
TIME_PROFILE = "evening_rush"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build_city(key):
    """Run the full extract → elevation → speeds → carbon pipeline for one city."""
    cfg  = CITIES[key]
    base = DATA_DIR / f"terrain_{key}"
    graph_path   = str(base) + ".graphml"
    elev_path    = str(base) + "_elev.graphml"
    speeds_path  = str(base) + "_speeds.graphml"
    carbon_path  = str(base) + "_carbon.graphml"

    if Path(carbon_path).exists():
        print(f"\n[{cfg['name']}] using cached graph at {Path(carbon_path).name}")
    else:
        print(f"\n=== Building {cfg['name']} ===")
        extract_subgraph(
            center=cfg["center"],
            save_path=graph_path,
            target_nodes=TARGET_NODES,
            radius_m=RADIUS_M,
            city_name=cfg["name"],
        )
        add_elevation_to_graph(graph_path=graph_path, save_path=elev_path)
        add_speeds_and_travel_times(graph_path=elev_path, save_path=speeds_path)
        calculate_carbon_weights(graph_path=speeds_path, save_path=carbon_path)

    G = ox.load_graphml(carbon_path)
    # GraphML stores numeric attributes as strings — convert once for speed
    for _, _, d in G.edges(data=True):
        for k in list(d.keys()):
            try:
                d[k] = float(d[k])
            except (ValueError, TypeError):
                pass
    return G


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def grade_stats(G):
    grades = np.array([abs(float(d.get("grade", 0.0))) for _, _, d in G.edges(data=True)])
    return {
        "mean_grade_pct":     grades.mean() * 100,
        "max_grade_pct":      grades.max()  * 100,
        "p95_grade_pct":      np.percentile(grades, 95) * 100,
        "pct_steep_3":        (grades > 0.03).mean() * 100,
        "pct_steep_5":        (grades > 0.05).mean() * 100,
    }


def path_carbon(G, path, carbon_key):
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        edge_data = G.get_edge_data(u, v)
        if edge_data is None:
            continue
        k0 = next(iter(edge_data))
        total += float(edge_data[k0].get(carbon_key, 0.0))
    return total


def path_time(G, path, time_key):
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        edge_data = G.get_edge_data(u, v)
        if edge_data is None:
            continue
        k0 = next(iter(edge_data))
        t = float(edge_data[k0].get(time_key, 0.0))
        total += t if t != float("inf") else 1000.0
    return total


def pareto_gap_analysis(G, num_trips=200):
    time_key   = f"travel_time_{TIME_PROFILE}"
    carbon_key = f"carbon_{VEHICLE}_{TIME_PROFILE}"
    nodes = list(G.nodes())

    random.seed(42)
    np.random.seed(42)

    rows = []
    attempts = 0
    while len(rows) < num_trips and attempts < num_trips * 20:
        attempts += 1
        src = random.choice(nodes)
        dst = random.choice(nodes)
        if src == dst:
            continue
        try:
            fast_path  = nx.shortest_path(G, src, dst, weight=time_key)
            green_path = nx.shortest_path(G, src, dst, weight=carbon_key)
        except nx.NetworkXNoPath:
            continue

        fc = path_carbon(G, fast_path,  carbon_key)
        gc = path_carbon(G, green_path, carbon_key)
        ft = path_time(G,   fast_path,  time_key)
        gt = path_time(G,   green_path, time_key)
        if fc == 0:
            continue
        rows.append({
            "fast_carbon":   fc,
            "green_carbon":  gc,
            "fast_time":     ft,
            "green_time":    gt,
            "carbon_saved_pct": (fc - gc) / fc * 100,
            "time_penalty_pct": (gt - ft) / ft * 100 if ft > 0 else 0,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def summarise(name, grade, pareto_df):
    saved = pareto_df["carbon_saved_pct"]
    pen   = pareto_df["time_penalty_pct"]
    return {
        "City": name,
        "Mean grade %":              f"{grade['mean_grade_pct']:.2f}",
        "P95 grade %":               f"{grade['p95_grade_pct']:.2f}",
        "Max grade %":               f"{grade['max_grade_pct']:.2f}",
        "% edges grade>5%":          f"{grade['pct_steep_5']:.1f}",
        "N trips evaluated":         len(pareto_df),
        "Mean carbon savings %":     f"{saved.mean():.2f}",
        "Max carbon savings %":      f"{saved.max():.2f}",
        "% identical paths":         f"{(saved < 0.01).mean()*100:.1f}",
        "Mean time penalty %":       f"{pen.mean():.2f}",
    }


def make_plot(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: histogram of per-trip carbon savings
    ax = axes[0]
    bins = np.arange(0, 31, 1)
    for key, res in results.items():
        saved = res["pareto"]["carbon_saved_pct"]
        ax.hist(saved, bins=bins, alpha=0.55,
                label=f"{CITIES[key]['label']}  (mean: {saved.mean():.2f}%)",
                color=CITIES[key]["color"], edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Per-trip carbon savings (Dijkstra greenest vs fastest, %)", fontsize=11)
    ax.set_ylabel("Number of trips", fontsize=11)
    ax.set_title("Distribution of Available Carbon Savings", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right: side-by-side metric bars
    ax = axes[1]
    labels      = [CITIES[k]["label"] for k in results]
    colors      = [CITIES[k]["color"] for k in results]
    mean_sav    = [results[k]["pareto"]["carbon_saved_pct"].mean() for k in results]
    max_sav     = [results[k]["pareto"]["carbon_saved_pct"].max()  for k in results]
    pct_ident   = [(results[k]["pareto"]["carbon_saved_pct"] < 0.01).mean() * 100 for k in results]
    grade_p95   = [results[k]["grade"]["p95_grade_pct"] for k in results]

    x = np.arange(len(labels))
    w = 0.2
    bars1 = ax.bar(x - 1.5*w, mean_sav,    w, label="Mean carbon savings %", color="#2ca02c")
    bars2 = ax.bar(x - 0.5*w, max_sav,     w, label="Max carbon savings %",  color="#ff7f0e")
    bars3 = ax.bar(x + 0.5*w, pct_ident,   w, label="% trips with identical fastest/greenest", color="#9467bd")
    bars4 = ax.bar(x + 1.5*w, grade_p95,   w, label="P95 |grade| %",         color="#8c564b")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_title("Carbon-Aware Routing Benefit vs Terrain", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    for bars in (bars1, bars2, bars3, bars4):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.1f}",
                    ha="center", va="bottom", fontsize=8)

    fig.suptitle("Terrain Study: Hilly Cities Have Larger Carbon-Aware Routing Benefit", fontsize=13)
    plt.tight_layout()
    out = RESULTS_DIR / "terrain_comparison.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"\nComparison plot saved → {out}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for key, cfg in CITIES.items():
        G       = build_city(key)
        grade   = grade_stats(G)
        print(f"\n[{cfg['name']}]")
        print(f"  {len(G.nodes)} nodes, {len(G.edges)} edges")
        print(f"  mean |grade|: {grade['mean_grade_pct']:.2f}%  "
              f"P95: {grade['p95_grade_pct']:.2f}%  "
              f"max: {grade['max_grade_pct']:.2f}%")
        print(f"  % steep edges (>5% grade): {grade['pct_steep_5']:.1f}%")

        print(f"  Running {NUM_TRIPS}-trip Dijkstra Pareto analysis...")
        df = pareto_gap_analysis(G, num_trips=NUM_TRIPS)
        saved = df["carbon_saved_pct"]
        print(f"  Mean carbon savings: {saved.mean():.2f}%")
        print(f"  Max  carbon savings: {saved.max():.2f}%")
        print(f"  Trips with identical paths: {(saved < 0.01).sum()}/{len(df)} "
              f"({(saved < 0.01).mean()*100:.1f}%)")
        results[key] = {"grade": grade, "pareto": df}

    # CSV summary
    summary = pd.DataFrame([summarise(CITIES[k]["name"], results[k]["grade"], results[k]["pareto"])
                            for k in results])
    csv_path = RESULTS_DIR / "terrain_comparison.csv"
    summary.to_csv(csv_path, index=False)
    print(f"\nSummary table saved → {csv_path}")
    print("\n" + summary.to_string(index=False))

    make_plot(results)


if __name__ == "__main__":
    main()
