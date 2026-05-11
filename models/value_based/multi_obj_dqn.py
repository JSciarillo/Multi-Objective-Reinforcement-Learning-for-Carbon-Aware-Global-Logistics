"""
Multi-Objective DQN — Carbon-Aware Routing
Extends the single-objective DQNto two objectives:
  1. Minimize CO2 emissions
  2. Minimize travel time
"""
import os
import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import osmnx as ox
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple

os.makedirs('morl', exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

#configs
GRAPHML_PATH = 'data/dc_subgraph_carbon.graphml'
CARBON_ATTR = 'carbon_HeavyTruck_evening_rush'
TIME_ATTR = 'travel_time_evening_rush'
N_OBJECTIVES  = 2  # carbon + time
N_EPISODES = 8000  # more episodes for multi-objective
MAX_STEPS = 30
BATCH_SIZE = 64
REPLAY_BUFFER = 5000
GAMMA = 0.99
LR = 5e-4
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
HIDDEN_DIM = 128
TARGET_UPDATE = 5
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
print("Config finished")


#50 node Subgraph
print("Loading graph...")
G_full = ox.load_graphml(GRAPHML_PATH)
nodes_full = list(G_full.nodes())

out_degrees = {n: len(list(G_full.successors(n))) for n in nodes_full}
center_node = max(out_degrees, key=out_degrees.get)

lengths = nx.single_source_shortest_path_length(G_full, center_node, cutoff=15)
nearby = sorted(lengths.keys(), key=lambda n: lengths[n])[:50]
G = G_full.subgraph(nearby).copy()
nodes = list(G.nodes())
print(f"  Subgraph: {len(nodes)} nodes, {len(G.edges())} edges")

reachable_pairs = [
    (o, d) for o in nodes for d in nodes
    if o != d and nx.has_path(G, o, d)
]
print(f"  Reachable O-D pairs: {len(reachable_pairs)}")

path_lengths = dict(nx.all_pairs_shortest_path_length(G))

all_carbon = [float(d.get(CARBON_ATTR, 0)) for _, _, d in G.edges(data=True)]
all_time = [float(d.get(TIME_ATTR,   0)) for _, _, d in G.edges(data=True)]
CARBON_MAX = max(all_carbon) if all_carbon else 1.0
TIME_MAX = max(all_time)   if all_time   else 1.0

lats = {n: float(G.nodes[n]['y']) for n in nodes}
lons = {n: float(G.nodes[n]['x']) for n in nodes}
lat_min = min(lats.values());  lat_max = max(lats.values())
lon_min = min(lons.values());  lon_max = max(lons.values())

def node_to_coords(n):
    lat = (lats[n] - lat_min) / (lat_max - lat_min + 1e-8)
    lon = (lons[n] - lon_min) / (lon_max - lon_min + 1e-8)
    return lat, lon

HIGHWAY_TYPES = {
    'motorway': 1.0, 'trunk': 0.9, 'primary': 0.8,
    'secondary': 0.6, 'tertiary': 0.4, 'residential': 0.2,
    'unclassified': 0.1, 'service': 0.05,
}
def highway_to_float(hw):
    if isinstance(hw, list): hw = hw[0]
    return HIGHWAY_TYPES.get(str(hw), 0.1)

print("Subgraph ready")

MAX_ACTIONS = max(len(list(G.successors(n))) for n in nodes)
N_EDGE_FEATS = 4   # carbon, time, grade, highway
STATE_DIM = 5   # cur_lat, cur_lon, dst_lat, dst_lon, steps_remaining

class RoutingEnv:
    """
    Multi-objective routing environment.

    State:   [cur_lat, cur_lon, dst_lat, dst_lon, steps_remaining/MAX_STEPS]
    Actions: up to MAX_ACTIONS neighbors, each represented by edge features
    Reward:  2D vector [−carbon + progress + cycle, −time + progress + cycle]
    """
    def __init__(self, G, nodes, reachable_pairs):
        self.G = G
        self.nodes = nodes
        self.reachable_pairs = reachable_pairs

    def reset(self, origin=None, destination=None):
        if origin is None or destination is None:
            self.origin, self.destination = random.choice(self.reachable_pairs)
        else:
            self.origin, self.destination = origin, destination
        self.current = self.origin
        self.steps = 0
        self.visited = {self.origin}
        return self._get_state()

    def _get_state(self):
        cur_lat, cur_lon = node_to_coords(self.current)
        dst_lat, dst_lon = node_to_coords(self.destination)
        return np.array([
            cur_lat, cur_lon,
            dst_lat, dst_lon,
            (MAX_STEPS - self.steps) / MAX_STEPS,
        ], dtype=np.float32)

    def get_action_features(self):
        neighbors = list(self.G.successors(self.current))
        features = np.zeros((MAX_ACTIONS, N_EDGE_FEATS), dtype=np.float32)
        for i, nbr in enumerate(neighbors[:MAX_ACTIONS]):
            ed = self.G.get_edge_data(self.current, nbr)
            if ed is None: continue
            best = ed[0] if 0 in ed else ed
            features[i] = [
                float(best.get(CARBON_ATTR, 0)) / CARBON_MAX,
                float(best.get(TIME_ATTR,   0)) / TIME_MAX,
                float(best.get('grade', 0)),
                highway_to_float(best.get('highway', 'unclassified')),
            ]
        return features, neighbors

    def step(self, action_idx):
        _, neighbors = self.get_action_features()

        if not neighbors or action_idx >= len(neighbors):
            return (self._get_state(),
                    np.zeros((MAX_ACTIONS, N_EDGE_FEATS), dtype=np.float32),
                    np.array([-2.0, -2.0], dtype=np.float32),
                    True, 'invalid')

        next_node = neighbors[action_idx]
        ed = self.G.get_edge_data(self.current, next_node)
        if ed is None:
            return (self._get_state(),
                    np.zeros((MAX_ACTIONS, N_EDGE_FEATS), dtype=np.float32),
                    np.array([-2.0, -2.0], dtype=np.float32),
                    True, 'no_edge')

        best = ed[0] if 0 in ed else ed
        carbon = float(best.get(CARBON_ATTR, 0)) / CARBON_MAX
        time = float(best.get(TIME_ATTR,   0)) / TIME_MAX

        # Progress reward
        try:
            progress = (path_lengths[self.current][self.destination] -
                        path_lengths[next_node][self.destination]) * 0.1
        except KeyError:
            progress = 0.0

        self.current = next_node
        self.steps += 1

        # Cycle penalty
        cycle = -0.05 if self.current in self.visited else 0.0
        self.visited.add(self.current)

        # 2D reward vector — one value per objective
        reward = np.array([
            -carbon + progress + cycle,
            -time   + progress + cycle,
        ], dtype=np.float32)

        done = False
        reason = 'step'

        if self.current == self.destination:
            reward += np.array([2.0, 2.0], dtype=np.float32)
            done, reason = True, 'reached'
        elif self.steps >= MAX_STEPS:
            reward -= np.array([1.0, 1.0], dtype=np.float32)
            done, reason = True, 'timeout'

        next_feat, _ = self.get_action_features()
        return self._get_state(), next_feat, reward, done, reason


env = RoutingEnv(G, nodes, reachable_pairs)

# Sanity check
print("Random agent")
reached = 0
for _ in range(200):
    env.reset()
    for _ in range(MAX_STEPS):
        nbrs = list(G.successors(env.current))
        if not nbrs: break
        _, _, _, done, reason = env.step(random.randint(0, len(nbrs) - 1))
        if reason == 'reached':
            reached += 1
            break
        if done: break
print(f" Random reach rate: {reached}/200")
print("Environment ready")


#Multi-objective DQN Network
class MoDQN(nn.Module):
    """
    Multi-Objective DQN with edge feature action representation.

    Same two-stream as the single-objective DQN,
    with two additions:
      1. Preference vector w = [w_carbon, w_time] where w_carbon + w_time = 1 concatenated to state input
      2. Output is (B, MAX_ACTIONS, N_OBJECTIVES) instead of (B, MAX_ACTIONS)
    """
    def __init__(self, state_dim, n_actions, n_edge_feats, n_objectives, hidden_dim):
        super().__init__()
        self.n_actions    = n_actions
        self.n_objectives = n_objectives

        #Context: state + preference = hidden
        self.context = nn.Sequential(
            nn.Linear(state_dim + n_objectives, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        #Action encoder: edge features = hidden//2
        self.action_enc = nn.Sequential(
            nn.Linear(n_edge_feats, hidden_dim // 2),
            nn.ReLU(),
        )

        #Score: context + action = Q-value per objective
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_objectives),  # one Q-value per objective
        )

    def forward(self, state, preference, act_features):
        """
        state:        (B, STATE_DIM)
        preference:   (B, N_OBJECTIVES)
        act_features: (B, MAX_ACTIONS, N_EDGE_FEATS)
        returns:      (B, MAX_ACTIONS, N_OBJECTIVES)
        """
        B = state.shape[0]

        #concat state with preference before encoding
        ctx_in = torch.cat([state, preference], dim=-1)
        ctx = self.context(ctx_in)
        ctx = ctx.unsqueeze(1).expand(-1, self.n_actions, -1)

        af = self.action_enc(act_features.view(B * self.n_actions, -1))
        af = af.view(B, self.n_actions, -1)

        combined = torch.cat([ctx, af], dim=-1)
        q = self.scorer(combined.view(B * self.n_actions, -1))
        return q.view(B, self.n_actions, self.n_objectives)

    def scalar_q(self, state, preference, act_features):
        """
        Scalarize Q-values using preference weights.
        Returns: (B, MAX_ACTIONS) - weighted sum across objectives
        """
        q = self.forward(state, preference, act_features)
        w = preference.unsqueeze(1)
        return (q * w).sum(dim=-1)


q_net = MoDQN(STATE_DIM, MAX_ACTIONS, N_EDGE_FEATS, N_OBJECTIVES, HIDDEN_DIM).to(DEVICE)
target_net = MoDQN(STATE_DIM, MAX_ACTIONS, N_EDGE_FEATS, N_OBJECTIVES, HIDDEN_DIM).to(DEVICE)
target_net.load_state_dict(q_net.state_dict())
target_net.eval()
optimizer = optim.Adam(q_net.parameters(), lr=LR)
print(f"  MoDQN parameters: {sum(p.numel() for p in q_net.parameters()):,}")
print("Multi-objective DQN ready")

# Replay Buffer
Transition = namedtuple('Transition',
    ['state', 'act_feat', 'preference', 'action', 'reward',
     'next_state', 'next_act_feat', 'done'])

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, *args):
        self.buffer.append(Transition(*args))
    def draw(self, n):
        return random.sample(self.buffer, n)
    def __len__(self):
        return len(self.buffer)

replay_buffer = ReplayBuffer(REPLAY_BUFFER)
print("Replay buffer ready")

# Training
def sample_preference():
    """Sample random [w_carbon, w_time] that sums to 1"""
    w = np.random.dirichlet([1, 1]).astype(np.float32)
    return torch.tensor(w).unsqueeze(0).to(DEVICE)

def select_action(state_t, pref_t, act_feat_t, epsilon):
    _, neighbors = env.get_action_features()
    n_valid = len(neighbors)
    if n_valid == 0:
        return 0
    if random.random() < epsilon:
        return random.randint(0, n_valid - 1)
    with torch.no_grad():
        sq   = q_net.scalar_q(state_t, pref_t, act_feat_t).squeeze(0)
        mask = torch.full((MAX_ACTIONS,), float('-inf')).to(DEVICE)
        mask[:n_valid] = 0
        return (sq + mask).argmax().item()

def train_step(pref_t):
    if len(replay_buffer) < BATCH_SIZE:
        return None

    batch = replay_buffer.draw(BATCH_SIZE)
    B = BATCH_SIZE
    states = torch.tensor(np.array([t.state         for t in batch])).to(DEVICE)
    act_feats = torch.tensor(np.array([t.act_feat      for t in batch])).to(DEVICE)
    prefs = torch.tensor(np.array([t.preference    for t in batch])).to(DEVICE)
    actions = torch.tensor([t.action for t in batch], dtype=torch.long).to(DEVICE)
    rewards = torch.tensor(np.array([t.reward        for t in batch])).to(DEVICE)
    next_states = torch.tensor(np.array([t.next_state    for t in batch])).to(DEVICE)
    next_feats = torch.tensor(np.array([t.next_act_feat for t in batch])).to(DEVICE)
    dones = torch.tensor([t.done  for t in batch], dtype=torch.float32).to(DEVICE)

    #Current Q-values-(B, O)
    q_all  = q_net(states, prefs, act_feats)
    q_curr = q_all[torch.arange(B), actions, :]

    #Target Q-values with Bellman equation
    with torch.no_grad():
        #sample new preferences for target computation
        target_prefs = torch.tensor(
            np.array([np.random.dirichlet([1, 1]).astype(np.float32)
                      for _ in range(B)])
        ).to(DEVICE)

        q_next_all = target_net(next_states, target_prefs, next_feats)
        sq_next = (q_next_all * target_prefs.unsqueeze(1)).sum(dim=-1)
        best_actions = sq_next.argmax(dim=-1)
        q_next = q_next_all[torch.arange(B), best_actions, :]
        target = rewards + GAMMA * q_next * (1 - dones.unsqueeze(1))

    loss = nn.functional.mse_loss(q_curr, target)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
    optimizer.step()
    return loss.item()


print("Starting Multi-Objective DQN training...")
print(f" Episodes: {N_EPISODES} | Nodes: {len(nodes)} | Device: {DEVICE}")
print(f" Objectives: carbon + time | Preference: Dirichlet sampled")
print()

epsilon = EPSILON_START
log = []

for episode in range(N_EPISODES):
    # Sample a preference for this episode
    pref = sample_preference()
    pref_np = pref.squeeze(0).cpu().numpy()

    state = env.reset()
    act_feat, _ = env.get_action_features()
    state_t = torch.tensor(state).unsqueeze(0).to(DEVICE)
    act_feat_t = torch.tensor(act_feat).unsqueeze(0).to(DEVICE)

    ep_reward = np.zeros(N_OBJECTIVES)
    ep_loss = []
    done = False
    reason = 'step'

    while not done:
        action = select_action(state_t, pref, act_feat_t, epsilon)
        next_state, next_feat, reward, done, reason = env.step(action)

        replay_buffer.push(
            state, act_feat, pref_np, action,
            reward.astype(np.float32),
            next_state, next_feat, float(done)
        )

        loss = train_step(pref)
        if loss is not None:
            ep_loss.append(loss)

        state = next_state
        act_feat = next_feat
        state_t = torch.tensor(state).unsqueeze(0).to(DEVICE)
        act_feat_t = torch.tensor(act_feat).unsqueeze(0).to(DEVICE)
        ep_reward += reward

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(q_net.state_dict())

    log.append({
        'episode': episode,
        'reward_carbon': ep_reward[0],
        'reward_time': ep_reward[1],
        'scalarized': (pref_np * ep_reward).sum(),
        'loss': np.mean(ep_loss) if ep_loss else 0,
        'epsilon': epsilon,
        'pref_carbon': pref_np[0],
        'pref_time': pref_np[1],
        'reached': reason == 'reached',
    })

    if episode % 500 == 0:
        recent = log[-100:] if len(log) >= 100 else log
        avg_reach = np.mean([r['reached']    for r in recent])
        avg_loss  = np.mean([r['loss']       for r in recent])
        print(f" Ep {episode:5d}/{N_EPISODES} | ε={epsilon:.3f} | loss={avg_loss:.4f} | reach={avg_reach:.1%}")

torch.save(q_net.state_dict(), 'morl/mo_dqn_model.pt')
df_log = pd.DataFrame(log)
df_log.to_csv('morl/mo_dqn_training_log.csv', index=False)
print()
print("Training complete")
print(f"  Final reach rate: {df_log['reached'].tail(500).mean():.1%}")


# Evaluate: Pareto Frontier
print("\ncreating Pareto frontier")

def run_rl_route(origin, destination, pref_np):
    """Run trained MoDQN agent with a fixed preference."""
    pref_t = torch.tensor(pref_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    state = env.reset(origin=origin, destination=destination)
    total_co2  = 0.0
    total_time = 0.0

    for _ in range(MAX_STEPS):
        act_feat, neighbors = env.get_action_features()
        n_valid = len(neighbors)
        if n_valid == 0: break

        state_t = torch.tensor(state).unsqueeze(0).to(DEVICE)
        act_feat_t = torch.tensor(act_feat).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            sq = q_net.scalar_q(state_t, pref_t, act_feat_t).squeeze(0)
            mask = torch.full((MAX_ACTIONS,), float('-inf')).to(DEVICE)
            mask[:n_valid] = 0
            action = (sq + mask).argmax().item()

        next_node  = neighbors[action]
        ed = G.get_edge_data(env.current, next_node)
        if ed is None: break
        best = ed[0] if 0 in ed else ed
        total_co2 += float(best.get(CARBON_ATTR, 0))
        total_time += float(best.get(TIME_ATTR,   0))

        state, _, _, done, _ = env.step(action)
        if done: break

    reached = (env.current == destination)
    return total_co2, total_time, reached


def run_optimal_route(origin, destination, attr):
    """Dijkstra minimizing a single attribute — same as baseline."""
    def weight_fn(u, v, data):
        best = data[0] if 0 in data else data
        return float(best.get(attr, 0))
    try:
        path = nx.dijkstra_path(G, origin, destination, weight=weight_fn)
        co2, time = 0.0, 0.0
        for u, v in zip(path[:-1], path[1:]):
            ed = G.get_edge_data(u, v)
            best = ed[0] if 0 in ed else ed
            co2 += float(best.get(CARBON_ATTR, 0))
            time += float(best.get(TIME_ATTR,   0))
        return co2, time, True
    except nx.NetworkXNoPath:
        return 0, 0, False


# Sweep preferences across O-D pairs to build Pareto frontier
N_OD_PAIRS = 20
N_PREF_SWEEP = 9

od_pairs = random.sample(reachable_pairs, min(N_OD_PAIRS, len(reachable_pairs)))
pareto_points = []

for o, d in od_pairs:
    for w_carbon in np.linspace(0, 1, N_PREF_SWEEP):
        pref = np.array([w_carbon, 1.0 - w_carbon], dtype=np.float32)
        co2, time, reached = run_rl_route(o, d, pref)
        if reached:
            pareto_points.append({
                'co2': co2,
                'time_min': time / 60,
                'w_carbon': w_carbon,
            })

df_pareto = pd.DataFrame(pareto_points)
print(f"  Collected {len(df_pareto)} Pareto points from {N_OD_PAIRS} O-D pairs")

#compare RL vs baselines on 100 pairs
results = []
eval_pairs = random.sample(reachable_pairs, min(100, len(reachable_pairs)))

for o, d in eval_pairs:
    # Use balanced preference for comparison
    pref = np.array([0.5, 0.5], dtype=np.float32)
    rl_co2,   rl_time,   rl_ok   = run_rl_route(o, d, pref)
    dijk_co2, dijk_time, dijk_ok = run_optimal_route(o, d, CARBON_ATTR)
    fast_co2, fast_time, fast_ok = run_optimal_route(o, d, TIME_ATTR)

    if rl_ok and dijk_ok and fast_ok:
        results.append({
            'rl_co2':    rl_co2,   'rl_time':   rl_time / 60,
            'dijk_co2':  dijk_co2, 'dijk_time': dijk_time / 60,
            'fast_co2':  fast_co2, 'fast_time': fast_time / 60,
        })

df_results = pd.DataFrame(results)
print(f"  Evaluated {len(df_results)} complete O-D pairs")

if len(df_results) > 0:
    co2_savings_rl = (df_results['fast_co2'] - df_results['rl_co2']).mean()
    co2_savings_dijk = (df_results['fast_co2'] - df_results['dijk_co2']).mean()
    pct_rl   = co2_savings_rl   / df_results['fast_co2'].mean() * 100
    pct_dijk = co2_savings_dijk / df_results['fast_co2'].mean() * 100

    print(f"\n  ── Route Comparison (balanced preference w=[0.5, 0.5]) ──")
    print(f"  {'Method':<22} {'Avg CO2 (g)':>12} {'Avg Time (min)':>15}")
    print(f"  {'─'*52}")
    print(f"  {'Fastest Path':<22} {df_results['fast_co2'].mean():>12.1f} {df_results['fast_time'].mean():>15.2f}")
    print(f"  {'MoDQN Agent (w=0.5)':<22} {df_results['rl_co2'].mean():>12.1f} {df_results['rl_time'].mean():>15.2f}")
    print(f"  {'Dijkstra (carbon)':<22} {df_results['dijk_co2'].mean():>12.1f} {df_results['dijk_time'].mean():>15.2f}")
    print(f"\n  CO2 savings vs fastest path:")
    print(f"    MoDQN Agent:       {co2_savings_rl:.1f}g ({pct_rl:.1f}%)")
    print(f"    Dijkstra (carbon): {co2_savings_dijk:.1f}g ({pct_dijk:.1f}%)")

df_results.to_csv('morl/mo_dqn_comparison.csv', index=False)


#training curves
fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor='k')
for ax in axes: ax.set_facecolor('#111111')

window = 100
df_log['scalarized_smooth'] = df_log['scalarized'].rolling(window).mean()
df_log['loss_smooth'] = df_log['loss'].rolling(window).mean()
df_log['reach_smooth'] = df_log['reached'].astype(float).rolling(window).mean()

axes[0].plot(df_log['episode'], df_log['scalarized_smooth'], color='#00c8ff', linewidth=1.5)
axes[0].set_title('Scalarized Reward (smoothed)', color='white')
axes[0].set_xlabel('Episode', color='white')
axes[0].tick_params(colors='white')

axes[1].plot(df_log['episode'], df_log['loss_smooth'], color='#ff6b35', linewidth=1.5)
axes[1].set_title('Training Loss (smoothed)', color='white')
axes[1].set_xlabel('Episode', color='white')
axes[1].tick_params(colors='white')

axes[2].plot(df_log['episode'], df_log['reach_smooth'], color='#00ff88', linewidth=1.5)
axes[2].set_title('Destination Reach Rate (smoothed)', color='white')
axes[2].set_xlabel('Episode', color='white')
axes[2].set_ylim(0, 1)
axes[2].tick_params(colors='white')

for ax in axes:
    for spine in ax.spines.values(): spine.set_edgecolor('#444444')

plt.suptitle('Multi-Objective DQN Training Curves (50-node Subgraph)', color='white', fontsize=14)
plt.tight_layout()
plt.savefig('morl/mo_dqn_curves.png', dpi=200, bbox_inches='tight', facecolor='k')
plt.show()
print("Training curves- morl/mo_dqn_curves.png")

# Pareto frontier
fig, ax = plt.subplots(figsize=(10, 6), facecolor='k')
ax.set_facecolor('k')

if len(df_pareto) > 0:
    scatter = ax.scatter(
        df_pareto['time_min'], df_pareto['co2'],
        c=df_pareto['w_carbon'], cmap='RdYlGn_r',
        s=80, alpha=0.8, edgecolors='white', linewidth=0.5,
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Carbon preference weight →', color='white', fontsize=10)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
else:
    ax.text(0.5, 0.5, 'No Pareto points collected\n(reach rate too low)',
            transform=ax.transAxes, color='white', ha='center', va='center', fontsize=14)

ax.set_xlabel('Travel Time (minutes)', color='white', fontsize=12)
ax.set_ylabel('CO₂ Emissions (g)', color='white', fontsize=12)
ax.set_title('Pareto Frontier — Speed vs. Emissions\n(Multi-Objective DQN, 50-node DC Subgraph)',
             color='white', fontsize=13)
ax.tick_params(colors='white')
for spine in ax.spines.values(): spine.set_edgecolor('white')

plt.tight_layout()
plt.savefig('morl/mo_dqn_pareto.png', dpi=200, bbox_inches='tight', facecolor='k')
plt.show()
print("Pareto frontier- morl/mo_dqn_pareto.png")

#Route comparison scatter
if len(df_results) > 0:
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='k')
    ax.set_facecolor('k')
    ax.scatter(df_results['fast_time'], df_results['fast_co2'],
               color='#ff4444', s=60, alpha=0.6, label='Fastest Path', zorder=3)
    ax.scatter(df_results['rl_time'], df_results['rl_co2'],
               color='#00c8ff', s=60, alpha=0.6, label='MoDQN Agent (w=0.5)', zorder=4)
    ax.scatter(df_results['dijk_time'], df_results['dijk_co2'],
               color='#00ff88', s=60, alpha=0.6, label='Dijkstra (carbon-optimal)', zorder=5)
    ax.set_xlabel('Travel Time (minutes)', color='white', fontsize=12)
    ax.set_ylabel('CO₂ Emissions (g)', color='white', fontsize=12)
    ax.set_title('Route Comparison: MoDQN Agent vs Baselines\n(50-node DC Subgraph, Heavy Truck Rush Hour)',
                 color='white', fontsize=13)
    ax.tick_params(colors='white')
    ax.legend(facecolor='#222222', labelcolor='white', fontsize=10)
    for spine in ax.spines.values(): spine.set_edgecolor('white')
    plt.tight_layout()
    plt.savefig('morl/mo_dqn_route_compare.png', dpi=200, bbox_inches='tight', facecolor='k')
    plt.show()
    print("Route comparison → morl/mo_dqn_route_compare.png")

print("\nFinal Summary")
print(f"  Device: {DEVICE}")
print(f"  Final reach: {df_log['reached'].tail(500).mean():.1%}")
print(f"  Pareto points: {len(df_pareto)}")
print(f"  O-D pairs eval: {len(df_results)}")
if len(df_results) > 0:
    print(f"  MoDQN CO2 savings: {pct_rl:.1f}% vs fastest path")
print(f"  Outputs in:     morl/")