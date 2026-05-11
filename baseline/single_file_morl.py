import argparse
import csv
import math
import os
import random
import sys
import time as _time
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------- Config ----------------
HERE        = os.path.dirname(os.path.abspath(__file__))
GRAPH_PATH  = os.path.join(HERE, "data", "dc_subgraph_carbon.graphml")
MODELS_DIR  = os.path.join(HERE, "models")
RESULTS_DIR = os.path.join(HERE, "results")
MODEL_PATH  = os.path.join(MODELS_DIR, "dqn_carbon.pt")
PARETO_PNG  = os.path.join(RESULTS_DIR, "pareto_frontier.png")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "summary.csv")

VEHICLE      = "HeavyTruck"
TIME_PROFILE = "evening_rush"
MAX_DEG      = 8        # max neighbors considered per node (= action space size)
SEED         = 0


# ---------------- Graph load ----------------
def load_simple_graph(path=GRAPH_PATH, vehicle=VEHICLE, time_profile=TIME_PROFILE):
    """Read the MultiDiGraph and collapse parallel edges to a simple DiGraph
    holding (time, carbon, length) per edge."""
    Gm = nx.read_graphml(path)
    G = nx.DiGraph()
    for n, d in Gm.nodes(data=True):
        G.add_node(n, x=float(d["x"]), y=float(d["y"]))
    t_key = f"travel_time_{time_profile}"
    c_key = f"carbon_{vehicle}_{time_profile}"
    for u, v, d in Gm.edges(data=True):
        try:
            t = float(d[t_key]); c = float(d[c_key])
        except (KeyError, ValueError, TypeError):
            continue
        length = float(d.get("length", 0.0))
        if G.has_edge(u, v):
            cur = G[u][v]
            # keep the cheapest parallel edge by joint (time+carbon)
            if t + c < cur["time"] + cur["carbon"]:
                cur.update(time=t, carbon=c, length=length)
        else:
            G.add_edge(u, v, time=t, carbon=c, length=length)
    if G.number_of_edges() == 0:
        raise RuntimeError("No usable edges - check VEHICLE / TIME_PROFILE keys.")
    # keep largest weakly connected component for sensible reachability
    wcc = max(nx.weakly_connected_components(G), key=len)
    return G.subgraph(wcc).copy()


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def path_cost(G, path):
    t = c = 0.0
    for u, v in zip(path, path[1:]):
        e = G[u][v]
        t += e["time"]; c += e["carbon"]
    return t, c


def compute_norms(G):
    """Per-edge median used so alpha=0.5 is unit-neutral between sec and grams."""
    times   = np.array([d["time"]   for _, _, d in G.edges(data=True)])
    carbons = np.array([d["carbon"] for _, _, d in G.edges(data=True)])
    return float(np.median(times)), float(np.median(carbons))


# ---------------- Baselines ----------------
def dijkstra_path(G, src, dst, weight):
    try:
        return nx.shortest_path(G, src, dst, weight=weight)
    except nx.NetworkXNoPath:
        return None


def weighted_astar(G, src, dst, alpha, T_norm, C_norm):
    """Weighted-A* with combined edge cost: alpha * t/T_norm + (1-alpha) * c/C_norm.
    Haversine heuristic (admissible lower bound on the time component only)."""
    if src == dst:
        return [src]
    dst_y, dst_x = G.nodes[dst]["y"], G.nodes[dst]["x"]
    def w_fn(u, v, e):
        return alpha * (e["time"] / T_norm) + (1 - alpha) * (e["carbon"] / C_norm)
    def h_fn(n, _goal):
        d_m  = haversine(G.nodes[n]["y"], G.nodes[n]["x"], dst_y, dst_x)
        t_lb = d_m / 30.0   # 30 m/s ~ 110 kph upper bound -> lower bound on time
        return alpha * (t_lb / T_norm)
    try:
        return nx.astar_path(G, src, dst, heuristic=h_fn, weight=w_fn)
    except nx.NetworkXNoPath:
        return None


# ---------------- Environment ----------------
class CarbonRoutingEnv:
    """Routing env with the *right* DQN architecture for variable action sets.

    Observation is a tuple (global_state, action_features, mask):
      - global_state:    (G_DIM,)          context that doesn't depend on the action
      - action_features: (MAX_DEG, A_DIM)  per-neighbor features (zeros for invalid slots)
      - mask:            (MAX_DEG,)        1.0 for valid neighbor slots

    The Q-network (see QNet below) computes Q(global, slot_features) -> scalar
    per slot, with weights *shared* across slots. This means action slot index
    is meaningless on its own; only the slot's feature vector matters, which
    fixes the generalization problem of a fixed-output-head DQN on a graph
    where slot 0 at node A is a totally different edge from slot 0 at node B.

    Additional ingredients:
      - Potential-based reward shaping: F = gamma*Phi(s') - Phi(s), Phi = -dist/D_norm.
      - Curriculum: reset(max_hops=k) restricts OD pairs to <= k graph hops.
    """

    # state layout constants
    G_DIM = 6         # cur_x, cur_y, dst_x, dst_y, dist_remaining_norm, alpha
    A_DIM = 6         # dx, dy, edge_t_norm, edge_c_norm, dist_after_norm, valid_flag

    def __init__(self, G, T_norm, C_norm, max_steps=300, gamma=0.98):
        self.G          = G
        self.nodes      = list(G.nodes())
        self.T_norm     = T_norm
        self.C_norm     = C_norm
        self.max_steps  = max_steps
        self.gamma      = gamma
        self.raw_neighbors = {n: list(G.successors(n))[:MAX_DEG] for n in self.nodes}
        self.coords     = {n: (G.nodes[n]["y"], G.nodes[n]["x"]) for n in self.nodes}
        ys = np.array([self.coords[n][0] for n in self.nodes])
        xs = np.array([self.coords[n][1] for n in self.nodes])
        self.y_min, self.y_max = ys.min(), ys.max()
        self.x_min, self.x_max = xs.min(), xs.max()
        self.dy_span = (self.y_max - self.y_min) or 1.0
        self.dx_span = (self.x_max - self.x_min) or 1.0
        self.D_norm = haversine(self.y_min, self.x_min, self.y_max, self.x_max) or 1.0

    def global_dim(self): return self.G_DIM
    def action_dim(self): return self.A_DIM

    def _norm_xy(self, n):
        y, x = self.coords[n]
        return (y - self.y_min) / self.dy_span, (x - self.x_min) / self.dx_span

    def _dist_to_dst(self, n):
        return haversine(*self.coords[n], *self.coords[self.dst])

    def _phi(self, n):
        return -self._dist_to_dst(n) / self.D_norm

    def _global_obs(self):
        cy, cx = self._norm_xy(self.cur)
        dy, dx = self._norm_xy(self.dst)
        d_rem  = self._dist_to_dst(self.cur) / self.D_norm
        return np.array([cx, cy, dx, dy, d_rem, self.alpha], dtype=np.float32)

    def _action_features(self):
        feats = np.zeros((MAX_DEG, self.A_DIM), dtype=np.float32)
        cur_y, cur_x = self.coords[self.cur]
        nbs = self.raw_neighbors[self.cur]
        for i in range(min(len(nbs), MAX_DEG)):
            v = nbs[i]
            vy, vx = self.coords[v]
            e = self.G[self.cur][v]
            feats[i, 0] = (vx - cur_x) / self.dx_span
            feats[i, 1] = (vy - cur_y) / self.dy_span
            feats[i, 2] = e["time"]   / self.T_norm
            feats[i, 3] = e["carbon"] / self.C_norm
            feats[i, 4] = self._dist_to_dst(v) / self.D_norm
            feats[i, 5] = 1.0
        return feats

    def _action_mask(self):
        nbs = self.raw_neighbors[self.cur]
        mask = np.zeros(MAX_DEG, dtype=np.float32)
        mask[:min(len(nbs), MAX_DEG)] = 1.0
        return mask

    def _obs(self):
        return self._global_obs(), self._action_features(), self._action_mask()

    def _sample_curriculum_pair(self, max_hops, rng, min_hops=2, max_tries=20):
        for _ in range(max_tries):
            s = rng.choice(self.nodes)
            if not self.raw_neighbors[s]:
                continue
            levels = {s: 0}
            frontier = [s]
            for depth in range(1, max_hops + 1):
                nxt_frontier = []
                for u in frontier:
                    for v in self.G.successors(u):
                        if v not in levels:
                            levels[v] = depth
                            nxt_frontier.append(v)
                frontier = nxt_frontier
                if not frontier:
                    break
            cands = [n for n, h in levels.items() if min_hops <= h <= max_hops]
            if cands:
                return s, rng.choice(cands)
        return None

    def reset(self, src=None, dst=None, alpha=None, rng=None, max_hops=None):
        rng = rng or random
        if src is None or dst is None:
            sd = None
            if max_hops is not None:
                sd = self._sample_curriculum_pair(max_hops=max_hops, rng=rng)
            if sd is None:
                for _ in range(50):
                    s = rng.choice(self.nodes); d = rng.choice(self.nodes)
                    if s != d and self.raw_neighbors[s] and nx.has_path(self.G, s, d):
                        sd = (s, d); break
            if sd is None:
                s = rng.choice([n for n in self.nodes if self.raw_neighbors[n]])
                d = rng.choice(self.raw_neighbors[s])
                sd = (s, d)
            src, dst = sd
        if alpha is None:
            alpha = rng.random()
        self.src, self.dst = src, dst
        self.alpha = float(alpha)
        self.cur = src
        self.steps = 0
        self.visited = {src}
        self.last_phi = self._phi(self.cur)
        return self._obs()

    # Convenience alias (rollout code uses this name).
    def action_mask(self):
        return self._action_mask()

    def step(self, a):
        nbs = self.raw_neighbors[self.cur]
        if a >= len(nbs):
            return self._obs(), -2.0, True, {"reason": "invalid"}
        nxt = nbs[a]
        e   = self.G[self.cur][nxt]
        r_time = -(e["time"]   / self.T_norm)
        r_carb = -(e["carbon"] / self.C_norm)
        r = self.alpha * r_time + (1 - self.alpha) * r_carb - 0.01
        if nxt in self.visited:
            r -= 0.3
        new_phi = self._phi(nxt)
        r += self.gamma * new_phi - self.last_phi
        self.last_phi = new_phi
        self.cur = nxt
        self.visited.add(nxt)
        self.steps += 1
        done, info = False, {}
        if self.cur == self.dst:
            r += 20.0; done = True; info["reason"] = "goal"
        elif self.steps >= self.max_steps:
            done = True; info["reason"] = "timeout"; r -= 2.0
        return self._obs(), float(r), done, info


# ---------------- Q-Network + Replay ----------------
class QNet(nn.Module):
    """Action-as-input Q-network: weights shared across slots, slot index ignored.

    Input  : global_state (B, G_DIM), action_features (B, MAX_DEG, A_DIM)
    Output : Q-values (B, MAX_DEG)
    """
    def __init__(self, global_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, global_state, action_features):
        B, K, _ = action_features.shape
        g = global_state.unsqueeze(1).expand(-1, K, -1)
        x = torch.cat([g, action_features], dim=-1)
        return self.net(x).squeeze(-1)


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buf = deque(maxlen=capacity)
    def push(self, *args):
        self.buf.append(args)
    def sample(self, batch_size):
        idx = np.random.choice(len(self.buf), batch_size, replace=False)
        batch = [self.buf[i] for i in idx]
        return list(zip(*batch))
    def __len__(self):
        return len(self.buf)


# ---------------- Training loop (curriculum + shaped reward + action-as-input) ----------------
def _act_greedy(q, g_state, a_feats, mask):
    """Pick argmax Q over valid actions. Returns int slot index or None if none valid."""
    if not np.any(mask > 0.5):
        return None
    with torch.no_grad():
        qv = q(torch.from_numpy(g_state).unsqueeze(0),
               torch.from_numpy(a_feats).unsqueeze(0)).squeeze(0).cpu().numpy()
    qv = np.where(mask > 0.5, qv, -1e9)
    return int(np.argmax(qv))


def train_dqn(steps=120_000, batch_size=128, gamma=0.98, lr=5e-4,
              eps_start=1.0, eps_end=0.05, eps_decay_steps=60_000,
              target_sync=1000, model_out=MODEL_PATH,
              curriculum_start=4, curriculum_max=30, curriculum_step=2,
              expand_threshold=0.65, expand_window=150):
    """
    Action-as-input Double-DQN trainer with:
      - Curriculum: reset(max_hops=k). Starts at curriculum_start hops; grows by
        curriculum_step once rolling success rate over `expand_window` episodes
        exceeds expand_threshold, capped at curriculum_max.
      - Potential-based reward shaping from the env (dense distance signal).
    """
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    print("Loading graph...")
    G = load_simple_graph()
    T_norm, C_norm = compute_norms(G)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges | "
          f"T_norm={T_norm:.2f}s  C_norm={C_norm:.2f}g")
    env = CarbonRoutingEnv(G, T_norm, C_norm, max_steps=300, gamma=gamma)

    g_dim, a_dim = env.global_dim(), env.action_dim()
    dev = torch.device("cpu")
    q   = QNet(g_dim, a_dim).to(dev)
    qt  = QNet(g_dim, a_dim).to(dev)
    qt.load_state_dict(q.state_dict())
    opt = optim.Adam(q.parameters(), lr=lr)
    buf = ReplayBuffer(100_000)

    max_hops_allowed = curriculum_start
    g_state, a_feats, mask = env.reset(max_hops=max_hops_allowed)
    ep_count = 0; successes = 0
    recent_success = deque(maxlen=expand_window)
    t0 = _time.time()

    for step in range(1, steps + 1):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * step / eps_decay_steps)
        valid = np.where(mask > 0.5)[0]
        if len(valid) == 0:
            g_state, a_feats, mask = env.reset(max_hops=max_hops_allowed); continue
        if random.random() < eps:
            a = int(random.choice(valid))
        else:
            a = _act_greedy(q, g_state, a_feats, mask)

        (g_next, a_next, m_next), r, done, info = env.step(a)
        buf.push(g_state, a_feats, mask, a, r, g_next, a_next, m_next, float(done))
        g_state, a_feats, mask = g_next, a_next, m_next

        if done:
            reached_goal = info.get("reason") == "goal"
            if reached_goal:
                successes += 1
            recent_success.append(1 if reached_goal else 0)
            ep_count += 1
            if (len(recent_success) >= expand_window
                    and (sum(recent_success) / len(recent_success)) > expand_threshold
                    and max_hops_allowed < curriculum_max):
                old = max_hops_allowed
                max_hops_allowed = min(curriculum_max, max_hops_allowed + curriculum_step)
                recent_success.clear()
                print(f"  >> curriculum: max_hops {old} -> {max_hops_allowed}  (step {step})")
            g_state, a_feats, mask = env.reset(max_hops=max_hops_allowed)

        if len(buf) >= batch_size:
            G_, AF, M, A, R, G2, AF2, M2, D = buf.sample(batch_size)
            G_  = torch.from_numpy(np.array(G_))
            AF  = torch.from_numpy(np.array(AF))
            M   = torch.from_numpy(np.array(M))
            A   = torch.tensor(A, dtype=torch.long)
            R   = torch.tensor(R, dtype=torch.float32)
            G2  = torch.from_numpy(np.array(G2))
            AF2 = torch.from_numpy(np.array(AF2))
            M2  = torch.from_numpy(np.array(M2))
            D   = torch.tensor(D, dtype=torch.float32)

            q_all  = q(G_, AF)
            q_pred = q_all.gather(1, A.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                qn_online = q(G2, AF2).masked_fill(M2 < 0.5, -1e9)
                a_star    = qn_online.argmax(dim=1, keepdim=True)
                qn_target = qt(G2, AF2).gather(1, a_star).squeeze(1)
                target    = R + gamma * (1 - D) * qn_target
            loss = nn.functional.smooth_l1_loss(q_pred, target)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 5.0)
            opt.step()

        if step % target_sync == 0:
            qt.load_state_dict(q.state_dict())

        if step % 2000 == 0:
            sr_recent = (sum(recent_success) / len(recent_success)) if recent_success else 0.0
            sr_total  = successes / max(ep_count, 1)
            print(f"step {step:6d}  eps={eps:.3f}  ep={ep_count:5d}  "
                  f"hops<={max_hops_allowed:2d}  success_recent={sr_recent:.2%}  "
                  f"success_total={sr_total:.2%}  buf={len(buf)}  t={_time.time()-t0:.1f}s")

    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save({"state_dict": q.state_dict(),
                "global_dim": g_dim, "action_dim": a_dim, "max_deg": MAX_DEG,
                "T_norm": T_norm, "C_norm": C_norm}, model_out)
    print(f"Saved model to {model_out}")


# ---------------- DQN rollout ----------------
@torch.no_grad()
def dqn_rollout(q, env, src, dst, alpha, max_steps=600):
    g_state, a_feats, mask = env.reset(src=src, dst=dst, alpha=alpha)
    path  = [env.cur]
    for _ in range(max_steps):
        if not np.any(mask > 0.5):
            return None
        qv = q(torch.from_numpy(g_state).unsqueeze(0),
               torch.from_numpy(a_feats).unsqueeze(0)).squeeze(0).cpu().numpy()
        qv = np.where(mask > 0.5, qv, -1e9)
        a = int(np.argmax(qv))
        (g_state, a_feats, mask), _, done, info = env.step(a)
        path.append(env.cur)
        if done:
            return path if info.get("reason") == "goal" else None
    return None


# ---------------- Eval set + sweep ----------------
def build_eval_set(G, n_pairs=20, min_hops=4, max_hops=16, seed=42):
    rng = random.Random(seed)
    nodes = list(G.nodes())
    pairs = []
    tries = 0
    while len(pairs) < n_pairs and tries < n_pairs * 400:
        tries += 1
        s = rng.choice(nodes); d = rng.choice(nodes)
        if s == d:
            continue
        try:
            sp_len = nx.shortest_path_length(G, s, d)
        except nx.NetworkXNoPath:
            continue
        if min_hops <= sp_len <= max_hops:
            pairs.append((s, d))
    return pairs


def eval_all(model_path=MODEL_PATH, n_pairs=20, alphas=None, run_dqn=True, run_astar=True):
    if alphas is None:
        alphas = [round(x, 2) for x in np.linspace(0.0, 1.0, 11)]
    G = load_simple_graph()
    T_norm, C_norm = compute_norms(G)
    env = CarbonRoutingEnv(G, T_norm, C_norm, max_steps=600)
    pairs = build_eval_set(G, n_pairs=n_pairs)
    print(f"Eval set: {len(pairs)} (src, dst) pairs, vehicle={VEHICLE}, time={TIME_PROFILE}")

    # Dijkstra endpoints (independent of alpha)
    dij_fast, dij_green = {}, {}
    for s, d in pairs:
        pf = dijkstra_path(G, s, d, weight="time")
        pg = dijkstra_path(G, s, d, weight="carbon")
        dij_fast[(s, d)]  = path_cost(G, pf) if pf else (None, None)
        dij_green[(s, d)] = path_cost(G, pg) if pg else (None, None)

    q_loaded = None
    if run_dqn and model_path and os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        # Back-compat: older checkpoints used (state_dim, max_deg) for a head-per-slot net.
        if "global_dim" in ckpt and "action_dim" in ckpt:
            q_loaded = QNet(ckpt["global_dim"], ckpt["action_dim"])
        else:
            raise RuntimeError("Old-format checkpoint detected. Re-run `train` to regenerate "
                               "models/dqn_carbon.pt with the new action-as-input architecture.")
        q_loaded.load_state_dict(ckpt["state_dict"])
        q_loaded.eval()
        print(f"Loaded DQN from {model_path}")
    elif run_dqn:
        print(f"No DQN checkpoint at {model_path} -- skipping DQN.")

    # Per-(alpha, pair) results: pair -> (t, c) or None
    astar_results = {alpha: {} for alpha in alphas}
    dqn_results   = {alpha: {} for alpha in alphas}
    for alpha in alphas:
        for s, d in pairs:
            if run_astar:
                pa = weighted_astar(G, s, d, alpha, T_norm, C_norm)
                astar_results[alpha][(s, d)] = path_cost(G, pa) if pa else None
            if q_loaded is not None:
                pd_ = dqn_rollout(q_loaded, env, s, d, alpha)
                dqn_results[alpha][(s, d)] = path_cost(G, pd_) if pd_ else None

    # Average each algorithm over ALL pairs it succeeded on (per-alpha).
    rows, astar_t, astar_c, dqn_t, dqn_c = [], [], [], [], []
    for alpha in alphas:
        a_ok = [v for v in astar_results[alpha].values() if v is not None]
        d_ok = [v for v in dqn_results[alpha].values()   if v is not None]
        a_t_mean = float(np.mean([v[0] for v in a_ok])) if a_ok else float("nan")
        a_c_mean = float(np.mean([v[1] for v in a_ok])) if a_ok else float("nan")
        d_t_mean = float(np.mean([v[0] for v in d_ok])) if d_ok else float("nan")
        d_c_mean = float(np.mean([v[1] for v in d_ok])) if d_ok else float("nan")
        astar_t.append(a_t_mean); astar_c.append(a_c_mean)
        dqn_t.append(d_t_mean);   dqn_c.append(d_c_mean)
        rows.append({
            "alpha": alpha,
            "astar_time_s": a_t_mean, "astar_carbon_g": a_c_mean,
            "astar_success": len(a_ok) / len(pairs) if run_astar else 0.0,
            "dqn_time_s":   d_t_mean, "dqn_carbon_g":   d_c_mean,
            "dqn_success":  (len(d_ok) / len(pairs)) if q_loaded is not None else 0.0,
        })
        print(f"alpha={alpha:>4.2f} | "
              f"A* t={a_t_mean:8.1f}s c={a_c_mean:9.1f}g succ={len(a_ok)}/{len(pairs)} | "
              f"DQN t={d_t_mean:8.1f}s c={d_c_mean:9.1f}g succ={len(d_ok)}/{len(pairs)}")

    fast_ts  = [dij_fast[p][0]  for p in pairs if dij_fast[p][0]  is not None]
    fast_cs  = [dij_fast[p][1]  for p in pairs if dij_fast[p][1]  is not None]
    green_ts = [dij_green[p][0] for p in pairs if dij_green[p][0] is not None]
    green_cs = [dij_green[p][1] for p in pairs if dij_green[p][1] is not None]
    fastest_t  = float(np.mean(fast_ts))  if fast_ts  else float("nan")
    fastest_c  = float(np.mean(fast_cs))  if fast_cs  else float("nan")
    greenest_t = float(np.mean(green_ts)) if green_ts else float("nan")
    greenest_c = float(np.mean(green_cs)) if green_cs else float("nan")
    print(f"\nDijkstra-fastest  (avg over {len(fast_ts)}/{len(pairs)} pairs): "
          f"{fastest_t:8.1f}s | {fastest_c:9.1f}g CO2")
    print(f"Dijkstra-greenest (avg over {len(green_ts)}/{len(pairs)} pairs): "
          f"{greenest_t:8.1f}s | {greenest_c:9.1f}g CO2")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(SUMMARY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Saved per-alpha summary to {SUMMARY_CSV}")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    if run_astar:
        ax.plot(astar_t, astar_c, "o-", label="Weighted A* (alpha sweep)", color="tab:blue")
    if q_loaded is not None and not all(math.isnan(x) for x in dqn_t):
        ax.plot(dqn_t, dqn_c, "s--", label="DQN (alpha sweep)", color="tab:orange")
    if not math.isnan(fastest_t):
        ax.scatter([fastest_t], [fastest_c], marker="*", s=180,
                   color="tab:green", label="Dijkstra-fastest", zorder=5)
    if not math.isnan(greenest_t):
        ax.scatter([greenest_t], [greenest_c], marker="*", s=180,
                   color="tab:red", label="Dijkstra-greenest", zorder=5)
    ax.set_xlabel("Avg travel time (s)")
    ax.set_ylabel("Avg CO2 emissions (g)")
    ax.set_title(f"Pareto frontier: time vs CO2 ({VEHICLE}, {TIME_PROFILE})")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(PARETO_PNG, dpi=140)
    print(f"Saved Pareto plot to {PARETO_PNG}")

    # ----- Matched per-pair comparison (DQN vs A*) on the subset where DQN succeeded -----
    if q_loaded is not None and run_astar:
        print("\n--- Matched comparison: DQN vs A* on the *same* OD pairs (only pairs where DQN succeeded) ---")
        for alpha in alphas:
            matched_pairs = [p for p in pairs
                             if dqn_results[alpha].get(p) is not None
                             and astar_results[alpha].get(p) is not None]
            if not matched_pairs:
                print(f"alpha={alpha:>4.2f}: DQN reached goal on 0 pairs (skipped).")
                continue
            a_t_m = np.mean([astar_results[alpha][p][0] for p in matched_pairs])
            a_c_m = np.mean([astar_results[alpha][p][1] for p in matched_pairs])
            d_t_m = np.mean([dqn_results[alpha][p][0]   for p in matched_pairs])
            d_c_m = np.mean([dqn_results[alpha][p][1]   for p in matched_pairs])
            t_gap = 100 * (d_t_m - a_t_m) / a_t_m
            c_gap = 100 * (d_c_m - a_c_m) / a_c_m
            print(f"alpha={alpha:>4.2f} | n={len(matched_pairs):2d}/{len(pairs)} | "
                  f"A* t={a_t_m:7.1f}s c={a_c_m:8.1f}g  vs  "
                  f"DQN t={d_t_m:7.1f}s c={d_c_m:8.1f}g  | "
                  f"DQN gap: time {t_gap:+6.1f}%  carbon {c_gap:+6.1f}%")

    if not math.isnan(fastest_t) and not math.isnan(fastest_c):
        print("\n--- Headline metrics (A* vs Dijkstra-fastest baseline, all pairs) ---")
        for alpha, t, c in zip(alphas, astar_t, astar_c):
            if math.isnan(t):
                continue
            t_pen = 100 * (t - fastest_t) / fastest_t
            c_red = 100 * (fastest_c - c) / fastest_c
            print(f"alpha={alpha:>4.2f}: time {t_pen:+6.1f}%  carbon {c_red:+6.1f}%")
        if q_loaded is not None:
            print("\n--- Headline metrics (DQN vs Dijkstra-fastest baseline) ---")
            for alpha, t, c in zip(alphas, dqn_t, dqn_c):
                if math.isnan(t):
                    continue
                t_pen = 100 * (t - fastest_t) / fastest_t
                c_red = 100 * (fastest_c - c) / fastest_c
                print(f"alpha={alpha:>4.2f}: time {t_pen:+6.1f}%  carbon {c_red:+6.1f}%")

    return rows


# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["train", "astar", "dqn", "compare"],
                    help="train | astar (baseline only) | dqn (trained only) | compare (all)")
    ap.add_argument("--steps", type=int, default=120_000, help="training steps (train mode)")
    ap.add_argument("--pairs", type=int, default=20,     help="eval OD pairs (eval modes)")
    args = ap.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.mode == "train":
        train_dqn(steps=args.steps)
    elif args.mode == "astar":
        eval_all(model_path=None, n_pairs=args.pairs, run_dqn=False, run_astar=True)
    elif args.mode == "dqn":
        if not os.path.exists(MODEL_PATH):
            sys.exit(f"No trained model at {MODEL_PATH}. Run `python single_file_morl.py train` first.")
        eval_all(model_path=MODEL_PATH, n_pairs=args.pairs, run_dqn=True, run_astar=False)
    elif args.mode == "compare":
        if not os.path.exists(MODEL_PATH):
            print(f"[warn] No model at {MODEL_PATH}; comparing A* + Dijkstra only.")
        eval_all(model_path=MODEL_PATH, n_pairs=args.pairs, run_dqn=True, run_astar=True)


if __name__ == "__main__":
    main()
