"""
Behavioral Cloning (BC) warm-up for PPO routing policies.

Instead of learning from random exploration, the policy first watches Dijkstra
solve thousands of trips and imitates its actions (supervised learning).
The BC-warmed model is then passed to train_ppo.py --use-bc for RL fine-tuning.

Why this helps:
  - Random PPO exploration on a 960-node graph rarely finds the destination
  - BC gives the policy a strong navigation prior in ~10 minutes
  - PPO fine-tuning then adjusts it toward carbon savings, not just speed

Usage:
    cd src
    python pretrain_bc.py                   # all 5 policies, 3000 episodes each
    python pretrain_bc.py --label balanced  # single policy
"""

import argparse
import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from torch.optim import Adam
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor

from rl_env import CarbonRoutingEnv

GRAPH_PATH = str(Path(__file__).parent.parent / "data" / "dc_subgraph_carbon.graphml")
MODELS_DIR = Path(__file__).parent.parent / "models"

PARETO_CONFIGS = [
    {"label": "fastest",       "alpha": 1.00, "beta": 0.00},
    {"label": "time_biased",   "alpha": 0.75, "beta": 0.25},
    {"label": "balanced",      "alpha": 0.50, "beta": 0.50},
    {"label": "carbon_biased", "alpha": 0.25, "beta": 0.75},
    {"label": "greenest",      "alpha": 0.00, "beta": 1.00},
]


# ---------------------------------------------------------------------------
# Expert data collection
# ---------------------------------------------------------------------------

def collect_expert_data(env, num_episodes=3000, alpha=1.0, beta=0.0):
    """
    Generate (state, action, mask) triples by following a Pareto-weighted Dijkstra
    expert. Each policy gets its OWN expert tailored to its (alpha, beta) weighting:
        weight(edge) = alpha * (time / time_norm) + beta * (carbon / carbon_norm)
    So the 'greenest' policy imitates carbon-minimising paths, the 'fastest' policy
    imitates time-minimising paths, and so on.
    """
    G = env.G
    time_key   = f"travel_time_{env.time_profile}"
    carbon_key = f"carbon_{env.vehicle}_{env.time_profile}"
    nodes = list(G.nodes())

    # GraphML attributes are stored as strings — convert to floats once
    for _, _, d in G.edges(data=True):
        for k in (time_key, carbon_key):
            if k in d:
                try:
                    d[k] = float(d[k])
                except (ValueError, TypeError):
                    d[k] = 10.0

    # Build the per-policy combined edge weight (Pareto scalarisation)
    pareto_key = f"_pareto_a{alpha}_b{beta}"
    for _, _, d in G.edges(data=True):
        t = float(d.get(time_key,   10.0))
        c = float(d.get(carbon_key, 10.0))
        if t == float("inf"):
            t = env.time_norm * 5
        d[pareto_key] = alpha * (t / env.time_norm) + beta * (c / env.carbon_norm)

    states, actions, masks = [], [], []
    successes = 0

    for _ in range(num_episodes * 4):
        if successes >= num_episodes:
            break

        start = random.choice(nodes)
        dest  = random.choice(nodes)
        if start == dest:
            continue

        try:
            path = nx.shortest_path(G, start, dest, weight=pareto_key)
        except nx.NetworkXNoPath:
            continue

        if len(path) < 2:
            continue

        obs, _ = env.reset(options={"start": start, "dest": dest})

        for i in range(len(path) - 1):
            curr = path[i]
            nxt  = path[i + 1]

            out_edges = list(G.out_edges(curr, keys=True))
            action_idx = None
            for j, (_, v, _) in enumerate(out_edges):
                if v == nxt:
                    action_idx = j
                    break

            if action_idx is None or action_idx >= env.max_actions:
                break

            # Build the valid-action mask for this state
            mask = np.zeros(env.max_actions, dtype=bool)
            for k in range(min(len(out_edges), env.max_actions)):
                mask[k] = True

            states.append(obs.copy())
            actions.append(action_idx)
            masks.append(mask.copy())

            obs, _, terminated, _, _ = env.step(action_idx)
            if terminated:
                break

        successes += 1
        if successes % 500 == 0:
            print(f"    {successes}/{num_episodes} episodes collected "
                  f"({len(states)} transitions so far)")

    print(f"  Collected {len(states)} transitions from {successes} expert episodes.")
    return (
        np.array(states,  dtype=np.float32),
        np.array(actions, dtype=np.int64),
        np.array(masks,   dtype=bool),
    )


# ---------------------------------------------------------------------------
# Behavioral cloning training loop
# ---------------------------------------------------------------------------

def run_bc(model, states, actions, masks, epochs=30, batch_size=128, lr=3e-4):
    """Maximise log-probability of expert actions (cross-entropy on policy logits)."""
    optimizer = Adam(model.policy.parameters(), lr=lr)
    idx = np.arange(len(states))

    best_loss = float("inf")

    for epoch in range(epochs):
        np.random.shuffle(idx)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, len(idx), batch_size):
            batch  = idx[start : start + batch_size]
            s_t    = torch.FloatTensor(states[batch])
            a_t    = torch.LongTensor(actions[batch])
            mask_t = torch.BoolTensor(masks[batch])

            # SB3 evaluate_actions returns (values, log_prob, entropy)
            _, log_prob, _ = model.policy.evaluate_actions(s_t, a_t, action_masks=mask_t)
            loss = -log_prob.mean()  # maximise log-prob of expert action

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg = epoch_loss / max(n_batches, 1)
        if avg < best_loss:
            best_loss = avg

        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1:>3}/{epochs}  loss: {avg:.4f}  best: {best_loss:.4f}")

    return model


# ---------------------------------------------------------------------------
# Per-policy entry point
# ---------------------------------------------------------------------------

def pretrain_single(cfg, graph_path=GRAPH_PATH, num_episodes=3000, bc_epochs=30):
    label = cfg["label"]
    alpha = cfg["alpha"]
    beta  = cfg["beta"]

    print(f"\n{'='*55}")
    print(f"  BC Pretraining: {label}  (α={alpha:.2f}, β={beta:.2f})")
    print(f"  Expert episodes: {num_episodes}   BC epochs: {bc_epochs}")
    print(f"{'='*55}")

    env = CarbonRoutingEnv(
        graph_path=graph_path,
        alpha=alpha, beta=beta,
        vehicle="HeavyTruck",
        time_profile="evening_rush",
        max_steps=500,
    )
    mon_env = Monitor(env)

    # Build a fresh model with the same architecture used in train_ppo.py
    model = MaskablePPO(
        "MlpPolicy", mon_env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=2048, batch_size=64, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.02,
        policy_kwargs={"net_arch": [256, 256]},
    )

    # Collect Pareto-weighted Dijkstra demonstrations and run BC
    states, actions, action_masks = collect_expert_data(
        env, num_episodes=num_episodes, alpha=alpha, beta=beta,
    )
    model = run_bc(model, states, actions, action_masks, epochs=bc_epochs)

    save_path = str(MODELS_DIR / f"ppo_{label}_bc")
    model.save(save_path)
    print(f"  Saved BC model → {save_path}.zip")
    return model


def pretrain_all(graph_path=GRAPH_PATH, num_episodes=3000, bc_epochs=30, label_filter=None):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    configs = PARETO_CONFIGS
    if label_filter:
        configs = [c for c in PARETO_CONFIGS if c["label"] == label_filter]
        if not configs:
            print(f"Unknown label. Choose from: {[c['label'] for c in PARETO_CONFIGS]}")
            sys.exit(1)

    for cfg in configs:
        pretrain_single(cfg, graph_path=graph_path,
                        num_episodes=num_episodes, bc_epochs=bc_epochs)

    print("\nBC pretraining done.")
    print("Next step: python train_ppo.py --use-bc --phase1 100000 --phase2 200000 --phase3 300000")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph",    default=GRAPH_PATH)
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--epochs",   type=int, default=30)
    parser.add_argument("--label",    default=None)
    args = parser.parse_args()

    pretrain_all(graph_path=args.graph, num_episodes=args.episodes,
                 bc_epochs=args.epochs, label_filter=args.label)
