"""
Trains one MaskablePPO agent per Pareto weight configuration using curriculum learning.

Curriculum phases (applied to every policy):
  Phase 1 — destinations within  8 hops  (100k steps) — easy, agent learns basic navigation
  Phase 2 — destinations within 20 hops  (150k steps) — medium trips
  Phase 3 — full graph, no hop limit      (250k steps) — real long-haul routing

Together the five saved models trace the Pareto frontier between fastest delivery
and lowest emissions.

Usage:
    cd src
    python train_ppo.py                    # full sweep, all 5 policies
    python train_ppo.py --label balanced   # single config
    python train_ppo.py --phase1 100000 --phase2 150000 --phase3 250000
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
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

CURRICULUM_PHASES = [
    {"hops":  8, "name": "Phase 1 (≤8 hops)"},
    {"hops": 20, "name": "Phase 2 (≤20 hops)"},
    {"hops": None, "name": "Phase 3 (full graph)"},
]


class SuccessRateLogger(BaseCallback):
    """Prints episode success rate every N episodes."""

    def __init__(self, print_every=50, verbose=0):
        super().__init__(verbose)
        self.print_every = print_every
        self._episode_successes = []
        self._n_episodes = 0

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._n_episodes += 1
                # SB3 Monitor adds 'episode' key; we detect success via step_count < max_steps
                # We tag success in the env info dict
                self._episode_successes.append(info.get("success", False))

                if self._n_episodes % self.print_every == 0:
                    recent = self._episode_successes[-self.print_every:]
                    rate = sum(recent) / len(recent)
                    print(f"    [ep {self._n_episodes:>5}] success rate (last {self.print_every}): "
                          f"{rate:.1%}  ({sum(recent)}/{len(recent)})")
        return True


def make_env(alpha, beta, curriculum_hops, graph_path):
    env = CarbonRoutingEnv(
        graph_path=graph_path,
        alpha=alpha,
        beta=beta,
        vehicle="HeavyTruck",
        time_profile="evening_rush",
        max_steps=500,
        curriculum_hops=curriculum_hops,
    )
    return Monitor(env)


def build_model(env, label, existing_model=None, use_bc=False):
    if existing_model is not None:
        existing_model.set_env(env)
        return existing_model

    # Load BC-pretrained weights if available and requested
    bc_path = MODELS_DIR / f"ppo_{label}_bc.zip"
    if use_bc and bc_path.exists():
        print(f"    Loading BC-pretrained weights from {bc_path.name}...")
        return MaskablePPO.load(
            str(bc_path), env=env,
            custom_objects={"learning_rate": 3e-4, "clip_range": 0.2},
        )

    return MaskablePPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        policy_kwargs={"net_arch": [256, 256]},
    )


def train_single(cfg, graph_path=GRAPH_PATH, phase_steps=(100_000, 150_000, 250_000), use_bc=False):
    label = cfg["label"]
    alpha = cfg["alpha"]
    beta  = cfg["beta"]

    print(f"\n{'='*55}")
    print(f"  Policy: {label}  (α={alpha:.2f}, β={beta:.2f})")
    print(f"  Phases: {phase_steps[0]:,} / {phase_steps[1]:,} / {phase_steps[2]:,} steps")
    if use_bc:
        print(f"  BC warm-up: enabled")
    print(f"{'='*55}")

    model = None
    for phase, (phase_cfg, steps) in enumerate(zip(CURRICULUM_PHASES, phase_steps), 1):
        hops = phase_cfg["hops"]
        name = phase_cfg["name"]
        print(f"\n  --- {name} ({steps:,} steps) ---")

        env = make_env(alpha, beta, hops, graph_path)
        model = build_model(env, label, model, use_bc=(use_bc and phase == 1))

        callbacks = [
            SuccessRateLogger(print_every=100),
            CheckpointCallback(
                save_freq=max(steps // 3, 10_000),
                save_path=str(MODELS_DIR / f"checkpoints_{label}"),
                name_prefix=f"ppo_phase{phase}",
            ),
        ]

        model.learn(
            total_timesteps=steps,
            callback=callbacks,
            reset_num_timesteps=(phase == 1),  # only reset timestep counter on phase 1
        )

    save_path = str(MODELS_DIR / f"ppo_{label}")
    model.save(save_path)
    print(f"\n  Saved → {save_path}.zip")
    return model


def train_pareto_sweep(
    graph_path=GRAPH_PATH,
    phase_steps=(100_000, 150_000, 250_000),
    label_filter=None,
    use_bc=False,
):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    configs = PARETO_CONFIGS
    if label_filter:
        configs = [c for c in PARETO_CONFIGS if c["label"] == label_filter]
        if not configs:
            valid = [c["label"] for c in PARETO_CONFIGS]
            print(f"Unknown label '{label_filter}'. Choose from: {valid}")
            sys.exit(1)

    for cfg in configs:
        train_single(cfg, graph_path=graph_path, phase_steps=phase_steps, use_bc=use_bc)

    total = sum(phase_steps) * len(configs)
    print(f"\nPareto sweep complete ({total:,} total timesteps across {len(configs)} policies).")
    print(f"Models saved to {MODELS_DIR}/ppo_*.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph",  default=GRAPH_PATH)
    parser.add_argument("--label",  default=None,
                        help="fastest | time_biased | balanced | carbon_biased | greenest")
    parser.add_argument("--phase1", type=int, default=100_000, help="Steps for phase 1 (≤8 hops)")
    parser.add_argument("--phase2", type=int, default=150_000, help="Steps for phase 2 (≤20 hops)")
    parser.add_argument("--phase3",  type=int, default=250_000, help="Steps for phase 3 (full graph)")
    parser.add_argument("--use-bc",  action="store_true",
                        help="Load BC-pretrained weights (run pretrain_bc.py first)")
    args = parser.parse_args()

    train_pareto_sweep(
        graph_path=args.graph,
        phase_steps=(args.phase1, args.phase2, args.phase3),
        label_filter=args.label,
        use_bc=args.use_bc,
    )
