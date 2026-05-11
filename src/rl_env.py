import gymnasium as gym
import numpy as np
import networkx as nx
import osmnx as ox
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
import os
from pathlib import Path
from collections import deque


class SimpleGCN(nn.Module):
    def __init__(self, in_features=3, out_features=64):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 32)
        self.fc2 = nn.Linear(32, out_features)

    def forward(self, X, A):
        H = F.relu(torch.sparse.mm(A, self.fc1(X)))
        return torch.sparse.mm(A, self.fc2(H))


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * np.arcsin(np.sqrt(a)) * 6_371_000  # meters


class CarbonRoutingEnv(gym.Env):
    """
    Multi-Objective RL environment for carbon-aware routing.
    Compatible with MaskablePPO from sb3-contrib via action_masks().

    State (dimensions assume max_actions=4):
        64  GCN embedding of current node
        64  GCN embedding of destination
         4  time-of-day one-hot
         2  compass bearing (delta_lon, delta_lat) to destination
        16  per-neighbour features (4 features x max_actions)
              for each outgoing edge: (delta_lon, delta_lat, time_norm, carbon_norm)
        ────
       150  total
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        graph_path,
        alpha=0.5,
        beta=0.5,
        vehicle="HeavyTruck",
        time_profile="evening_rush",
        max_steps=500,
        curriculum_hops=None,   # None = full graph; int = max BFS hops from start to dest
    ):
        super().__init__()

        graph_path = str(Path(graph_path).resolve())
        print(f"Loading graph from {graph_path}...")
        self.G = ox.load_graphml(graph_path)
        self.nodes = list(self.G.nodes())
        self.graph_path = graph_path

        self.alpha = alpha
        self.beta = beta
        self.vehicle = vehicle
        self.time_profile = time_profile
        self.max_steps = max_steps

        self.time_profiles = ["free_flow_midnight", "morning_rush", "lunch_traffic", "evening_rush"]
        self.time_idx = self.time_profiles.index(time_profile) if time_profile in self.time_profiles else 3
        self.curriculum_hops = curriculum_hops  # mutable — train_ppo updates this during training

        self.max_actions = max((self.G.out_degree(n) for n in self.nodes), default=1)

        # Normalisation constants from graph statistics (95th percentile keeps outliers from dominating)
        time_key = f"travel_time_{time_profile}"
        carbon_key = f"carbon_{vehicle}_{time_profile}"
        times = [
            float(d.get(time_key, 10.0))
            for _, _, d in self.G.edges(data=True)
            if d.get(time_key, 10.0) != float("inf")
        ]
        carbons = [float(d.get(carbon_key, 10.0)) for _, _, d in self.G.edges(data=True)]
        self.time_norm = float(np.percentile(times, 95)) if times else 100.0
        self.carbon_norm = float(np.percentile(carbons, 95)) if carbons else 500.0

        # Precompute lon/lat normalisation scale (typical DC degree span ≈ 0.1°)
        all_x = [float(self.G.nodes[n].get("x", 0.0)) for n in self.nodes]
        all_y = [float(self.G.nodes[n].get("y", 0.0)) for n in self.nodes]
        self._coord_scale = max(max(all_x) - min(all_x), max(all_y) - min(all_y), 1e-6)

        # Per-neighbour features: (delta_lon, delta_lat, time_norm, carbon_norm)
        self._neighbor_feat_dim = 4
        state_dim = 64 + 64 + 4 + 2 + self.max_actions * self._neighbor_feat_dim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                             shape=(state_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.max_actions)

        self.embeddings = self._load_or_compute_embeddings()

        # Runtime state (initialised in reset)
        self.current_node = self.nodes[0]
        self.dest_node = self.nodes[0]
        self.start_node = self.nodes[0]
        self.step_count = 0
        self.prev_dist = 1.0
        self.initial_dist = 1.0
        self.visited_nodes = set()

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _cache_path(self):
        mtime = os.path.getmtime(self.graph_path)
        h = hashlib.md5(f"{self.graph_path}_{mtime}".encode()).hexdigest()[:12]
        return Path(self.graph_path).parent / f"embeddings_{h}.npy"

    def _load_or_compute_embeddings(self):
        cache = self._cache_path()
        if cache.exists():
            print(f"Loading cached embeddings from {cache.name}...")
            Z = np.load(str(cache))
            return {n: Z[i] for i, n in enumerate(self.nodes)}

        print("Computing GNN embeddings (first run; will be cached)...")
        N = len(self.nodes)
        node_to_idx = {n: i for i, n in enumerate(self.nodes)}

        # Node features: lon, lat, elevation
        X = np.zeros((N, 3), dtype=np.float32)
        for i, n in enumerate(self.nodes):
            d = self.G.nodes[n]
            X[i] = [float(d.get("x", 0.0)), float(d.get("y", 0.0)), float(d.get("elevation", 0.0))]
        mu, sigma = X.mean(0), X.std(0)
        sigma[sigma == 0] = 1.0
        X = torch.FloatTensor((X - mu) / sigma)

        # Sparse symmetric adjacency with self-loops
        raw_edges = list(self.G.edges(keys=False))
        row = [node_to_idx[u] for u, v in raw_edges] + [node_to_idx[v] for u, v in raw_edges] + list(range(N))
        col = [node_to_idx[v] for u, v in raw_edges] + [node_to_idx[u] for u, v in raw_edges] + list(range(N))
        A = torch.sparse_coo_tensor(
            torch.LongTensor([row, col]),
            torch.ones(len(row)),
            (N, N),
        ).coalesce()

        torch.manual_seed(42)
        model = SimpleGCN(in_features=3, out_features=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        for _ in range(100):
            optimizer.zero_grad()
            Z = model(X, A)
            src, dst = A.indices()
            pos_loss = -F.logsigmoid((Z[src] * Z[dst]).sum(1)).mean()
            neg_dst = torch.randint(0, N, (len(src),))
            neg_loss = -F.logsigmoid(-(Z[src] * Z[neg_dst]).sum(1)).mean()
            (pos_loss + neg_loss).backward()
            optimizer.step()

        with torch.no_grad():
            Z = model(X, A).numpy()

        norms = np.linalg.norm(Z, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        Z = Z / norms

        np.save(str(cache), Z)
        print(f"Embeddings cached to {cache.name}")
        return {n: Z[i] for i, n in enumerate(self.nodes)}

    # ------------------------------------------------------------------
    # Core Gymnasium API
    # ------------------------------------------------------------------

    def _bfs_destinations(self, start, max_hops):
        """Return all nodes reachable from start in exactly 1..max_hops directed hops."""
        reachable = []
        visited = {start}
        frontier = [start]
        for _ in range(max_hops):
            next_frontier = []
            for node in frontier:
                for _, nb in self.G.out_edges(node):
                    if nb not in visited:
                        visited.add(nb)
                        next_frontier.append(nb)
                        reachable.append(nb)
            frontier = next_frontier
            if not frontier:
                break
        return reachable

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        if options and "start" in options and "dest" in options:
            self.start_node = options["start"]
            self.dest_node = options["dest"]
        elif self.curriculum_hops is not None:
            # Curriculum mode: destination is within curriculum_hops directed edges of start
            for _ in range(100):
                self.start_node = rng.choice(self.nodes)
                candidates = self._bfs_destinations(self.start_node, self.curriculum_hops)
                if candidates:
                    self.dest_node = rng.choice(candidates)
                    break
            else:
                # Fallback if BFS keeps coming up empty
                self.start_node = rng.choice(self.nodes)
                self.dest_node = rng.choice(self.nodes)
        else:
            while True:
                self.start_node = rng.choice(self.nodes)
                self.dest_node = rng.choice(self.nodes)
                if self.start_node == self.dest_node:
                    continue
                try:
                    nx.shortest_path(self.G, self.start_node, self.dest_node)
                    break
                except nx.NetworkXNoPath:
                    continue

        self.current_node = self.start_node
        self.step_count = 0
        self.visited_nodes = {self.start_node}
        self.initial_dist = haversine(
            self.G.nodes[self.start_node]["x"], self.G.nodes[self.start_node]["y"],
            self.G.nodes[self.dest_node]["x"], self.G.nodes[self.dest_node]["y"],
        )
        if self.initial_dist < 1.0:
            self.initial_dist = 1.0
        self.prev_dist = self.initial_dist

        return self._get_obs(), self._get_info()

    def step(self, action):
        self.step_count += 1
        out_edges = list(self.G.out_edges(self.current_node, keys=True, data=True))

        if action >= len(out_edges):
            return self._get_obs(), -1.0, False, self.step_count >= self.max_steps, self._get_info()

        _, v, _, data = out_edges[action]
        self.current_node = v

        time_key = f"travel_time_{self.time_profile}"
        carbon_key = f"carbon_{self.vehicle}_{self.time_profile}"

        t = float(data.get(time_key, 10.0))
        c = float(data.get(carbon_key, 10.0))
        if t == float("inf"):
            t = self.time_norm * 5

        norm_t = t / self.time_norm
        norm_c = c / self.carbon_norm

        curr_dist = haversine(
            self.G.nodes[self.current_node]["x"], self.G.nodes[self.current_node]["y"],
            self.G.nodes[self.dest_node]["x"], self.G.nodes[self.dest_node]["y"],
        )
        progress = (self.prev_dist - curr_dist) / self.initial_dist
        self.prev_dist = curr_dist

        # Heavy penalty for revisiting a node — breaks A→B→A cycles
        revisit_penalty = -1.0 if self.current_node in self.visited_nodes else 0.0
        self.visited_nodes.add(self.current_node)

        reward = -(self.alpha * norm_t) - (self.beta * norm_c) + progress * 0.5 + revisit_penalty

        terminated = self.current_node == self.dest_node
        truncated = self.step_count >= self.max_steps

        if terminated:
            reward += 30.0

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def action_masks(self):
        """Boolean mask over the fixed action space; required by MaskablePPO."""
        out_edges = list(self.G.out_edges(self.current_node, keys=True))
        mask = np.zeros(self.max_actions, dtype=bool)
        for i in range(min(len(out_edges), self.max_actions)):
            mask[i] = True
        return mask

    def _get_obs(self):
        time_vec = np.zeros(4, dtype=np.float32)
        time_vec[self.time_idx] = 1.0

        # Compass bearing: normalised (delta_lon, delta_lat) toward destination
        curr_x = float(self.G.nodes[self.current_node]["x"])
        curr_y = float(self.G.nodes[self.current_node]["y"])
        dest_x = float(self.G.nodes[self.dest_node]["x"])
        dest_y = float(self.G.nodes[self.dest_node]["y"])
        compass = np.array(
            [(dest_x - curr_x) / self._coord_scale,
             (dest_y - curr_y) / self._coord_scale],
            dtype=np.float32,
        )

        # Per-neighbour features: tells the policy what each action does.
        #   For action k: (Δlon, Δlat to neighbour, normalised edge time, normalised edge carbon)
        # Invalid actions get zero features and are masked out by action_masks().
        time_key   = f"travel_time_{self.time_profile}"
        carbon_key = f"carbon_{self.vehicle}_{self.time_profile}"
        out_edges = list(self.G.out_edges(self.current_node, keys=True, data=True))
        neighbor_features = np.zeros((self.max_actions, self._neighbor_feat_dim), dtype=np.float32)

        for i in range(min(len(out_edges), self.max_actions)):
            _, v, _, data = out_edges[i]
            v_x = float(self.G.nodes[v]["x"])
            v_y = float(self.G.nodes[v]["y"])
            t = float(data.get(time_key, 10.0))
            c = float(data.get(carbon_key, 10.0))
            if t == float("inf"):
                t = self.time_norm * 5
            neighbor_features[i, 0] = (v_x - curr_x) / self._coord_scale
            neighbor_features[i, 1] = (v_y - curr_y) / self._coord_scale
            neighbor_features[i, 2] = t / self.time_norm
            neighbor_features[i, 3] = c / self.carbon_norm

        return np.concatenate(
            [self.embeddings[self.current_node], self.embeddings[self.dest_node],
             time_vec, compass, neighbor_features.flatten()],
            dtype=np.float32,
        )

    def _get_info(self):
        return {
            "current_node": self.current_node,
            "dest_node": self.dest_node,
            "step_count": self.step_count,
            "success": self.current_node == self.dest_node,
        }
