"""
Carbon aware routing using DQN on a 50-node subgraph of DC

Objective is to minimize CO2 emissions
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

GRAPHML_PATH  = 'data/dc_subgraph_carbon.graphml'
CARBON_ATTR = 'carbon_HeavyTruck_evening_rush'
TIME_ATTR = 'travel_time_evening_rush'
N_EPISODES = 5000
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

# subgraph
print("fetching full graph")
G_full = ox.load_graphml(GRAPHML_PATH)
nodes_full = list(G_full.nodes())

#find center node from the max out degree
out_degrees = {n: len(list(G_full.successors(n))) for n in nodes_full}
center_node = max(out_degrees, key=out_degrees.get)

#50 nearest nodes
lengths = nx.single_source_shortest_path_length(G_full, center_node, cutoff=15)
nearby  = sorted(lengths.keys(), key=lambda n: lengths[n])[:50]

#subgraph formation
G = G_full.subgraph(nearby).copy()
nodes = list(G.nodes())
print(f"  Subgraph: {len(nodes)} nodes, {len(G.edges())} edges")

#precompute all reachable O-D pairs
reachable_pairs = [
    (o, d) for o in nodes for d in nodes
    if o != d and nx.has_path(G, o, d)
]
print(f" Reachable O-D pairs: {len(reachable_pairs)}")

#precompute shortest path lengths
path_lengths = dict(nx.all_pairs_shortest_path_length(G))

#normalize edge weights
all_carbon = [float(d.get(CARBON_ATTR, 0)) for _, _, d in G.edges(data=True)]
all_time = [float(d.get(TIME_ATTR, 0))   for _, _, d in G.edges(data=True)]
CARBON_MAX = max(all_carbon) if all_carbon else 1.0
TIME_MAX = max(all_time)   if all_time   else 1.0

#spatial coordinates
lats    = {n: float(G.nodes[n]['y']) for n in nodes}
lons    = {n: float(G.nodes[n]['x']) for n in nodes}
lat_min = min(lats.values());  lat_max = max(lats.values())
lon_min = min(lons.values());  lon_max = max(lons.values())

def node_to_coords(n):
    lat = (lats[n] - lat_min) / (lat_max - lat_min + 1e-8)
    lon = (lons[n] - lon_min) / (lon_max - lon_min + 1e-8)
    return lat, lon

#highway encoding
HIGHWAY_TYPES = {
    'motorway': 1.0, 'trunk': 0.9, 'primary': 0.8,
    'secondary': 0.6, 'tertiary': 0.4, 'residential': 0.2,
    'unclassified': 0.1, 'service': 0.05,
}
def highway_to_float(hw):
    if isinstance(hw, list): hw = hw[0]
    return HIGHWAY_TYPES.get(str(hw), 0.1)

print("Subgraph ready")

MAX_ACTIONS  = max(len(list(G.successors(n))) for n in nodes)
N_EDGE_FEATS = 4  # carbon, time, grade, highway
STATE_DIM    = 5   # cur_lat, cur_lon, dst_lat, dst_lon, steps_remaining

print(f"  Max actions: {MAX_ACTIONS}")
print(f"  State dim: {STATE_DIM}")

class RoutingEnv:
    def __init__(self, G, nodes, reachable_pairs):
        self.G = G
        self.nodes  = nodes
        self.reachable_pairs = reachable_pairs

    def reset(self, origin=None, destination=None):
        if origin is None or destination is None:
            self.origin, self.destination = random.choice(self.reachable_pairs)
        else:
            self.origin = origin
            self.destination = destination
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
        features  = np.zeros((MAX_ACTIONS, N_EDGE_FEATS), dtype=np.float32)
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
            return self._get_state(), np.zeros((MAX_ACTIONS, N_EDGE_FEATS), dtype=np.float32), -2.0, True, 'invalid'

        next_node = neighbors[action_idx]
        ed = self.G.get_edge_data(self.current, next_node)
        if ed is None:
            return self._get_state(), np.zeros((MAX_ACTIONS, N_EDGE_FEATS), dtype=np.float32), -2.0, True, 'no_edge'

        best = ed[0] if 0 in ed else ed
        carbon = float(best.get(CARBON_ATTR, 0)) / CARBON_MAX
        time = float(best.get(TIME_ATTR,   0)) / TIME_MAX

        #progress reward
        try:
            progress = (path_lengths[self.current][self.destination] -
                       path_lengths[next_node][self.destination]) * 0.1
        except KeyError:
            progress = 0.0

        self.current = next_node
        self.steps += 1

        #cycle penalty
        cycle = -0.05 if self.current in self.visited else 0.0
        self.visited.add(self.current)

        #minimize carbon
        reward = -carbon + progress + cycle

        done = False
        reason = 'step'

        if self.current == self.destination:
            reward += 2.0   # arrival bonus
            done = True
            reason = 'reached'
        elif self.steps >= MAX_STEPS:
            reward -= 1.0
            done = True
            reason = 'timeout'

        next_feat, _ = self.get_action_features()
        return self._get_state(), next_feat, reward, done, reason


env = RoutingEnv(G, nodes, reachable_pairs)

print("Random agent:")
reached = 0
for _ in range(200):
    env.reset()
    for _ in range(MAX_STEPS):
        nbrs = list(G.successors(env.current))
        if not nbrs:
            break
        _, _, _, done, reason = env.step(random.randint(0, len(nbrs) - 1))
        if reason == 'reached':
            reached += 1
            break
        if done:
            break
print(f"  Random reach rate: {reached}/200")
print("Environment ready")


#DQN Network
class DQN(nn.Module):
    """
    Simple DQN with edge feature action representation.
    Single objective: minimize carbon emissions.
    """
    def __init__(self, state_dim, n_actions, n_edge_feats, hidden_dim):
        super().__init__()
        self.n_actions = n_actions

        # Context: state = hidden
        self.context = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Action encoder: edge features = hidden//2
        self.action_enc = nn.Sequential(
            nn.Linear(n_edge_feats, hidden_dim // 2),
            nn.ReLU(),
        )

        # Scorer: context + action = Q-value
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, act_features):
        """
        state- (B, STATE_DIM)
        act_features- (B, MAX_ACTIONS, N_EDGE_FEATS)
        returns- (B, MAX_ACTIONS)
        """
        B = state.shape[0]
        ctx = self.context(state)                         
        ctx = ctx.unsqueeze(1).expand(-1, self.n_actions, -1)  

        af = act_features.view(B * self.n_actions, -1)
        af = self.action_enc(af).view(B, self.n_actions, -1) 

        combined = torch.cat([ctx, af], dim=-1)
        q = self.scorer(combined.view(B * self.n_actions, -1))
        return q.view(B, self.n_actions)


q_net = DQN(STATE_DIM, MAX_ACTIONS, N_EDGE_FEATS, HIDDEN_DIM).to(DEVICE)
target_net = DQN(STATE_DIM, MAX_ACTIONS, N_EDGE_FEATS, HIDDEN_DIM).to(DEVICE)
target_net.load_state_dict(q_net.state_dict())
target_net.eval()

optimizer = optim.Adam(q_net.parameters(), lr=LR)
print(f"DQN parameters: {sum(p.numel() for p in q_net.parameters()):,}")
print("DQN ready")


Transition = namedtuple('Transition',
    ['state', 'act_feat', 'action', 'reward',
     'next_state', 'next_act_feat', 'done'])

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, *args):
        self.buffer.append(Transition(*args))
    def sample(self, n):
        return random.sample(self.buffer, n)
    def __len__(self):
        return len(self.buffer)

replay_buffer = ReplayBuffer(REPLAY_BUFFER)
print("Replay buffer ready")


def select_action(state_t, act_feat_t, epsilon):
    nbrs = list(G.successors(env.current))
    n_valid = len(nbrs)
    if n_valid == 0:
        return 0
    if random.random() < epsilon:
        return random.randint(0, n_valid - 1)
    with torch.no_grad():
        q = q_net(state_t, act_feat_t).squeeze(0)
        mask = torch.full((MAX_ACTIONS,), float('-inf')).to(DEVICE)
        mask[:n_valid] = 0
        return (q + mask).argmax().item()

def train_step():
    if len(replay_buffer) < BATCH_SIZE:
        return None
    batch = replay_buffer.sample(BATCH_SIZE)

    states = torch.tensor(np.array([t.state for t in batch])).to(DEVICE)
    act_feats = torch.tensor(np.array([t.act_feat for t in batch])).to(DEVICE)
    actions = torch.tensor([t.action for t in batch], dtype=torch.long).to(DEVICE)
    rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32).to(DEVICE)
    next_states = torch.tensor(np.array([t.next_state for t in batch])).to(DEVICE)
    next_feats = torch.tensor(np.array([t.next_act_feat for t in batch])).to(DEVICE)
    dones = torch.tensor([t.done  for t in batch], dtype=torch.float32).to(DEVICE)

    q_curr = q_net(states, act_feats).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        q_next = target_net(next_states, next_feats).max(dim=1)[0]
        target = rewards + GAMMA * q_next * (1 - dones)

    loss = nn.functional.mse_loss(q_curr, target)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
    optimizer.step()
    return loss.item()


print("DQN training in progress")
print(f" Episodes: {N_EPISODES} | Subgraph: {len(nodes)} nodes | Device: {DEVICE}")
print()

epsilon = EPSILON_START
log = []

for episode in range(N_EPISODES):
    state = env.reset()
    act_feat, _  = env.get_action_features()
    state_t = torch.tensor(state).unsqueeze(0).to(DEVICE)
    act_feat_t = torch.tensor(act_feat).unsqueeze(0).to(DEVICE)

    ep_reward = 0.0
    ep_loss = []
    done = False
    reason = 'step'

    while not done:
        action = select_action(state_t, act_feat_t, epsilon)
        next_state, next_feat, reward, done, reason = env.step(action)

        replay_buffer.push(
            state, act_feat, action, reward,
            next_state, next_feat, float(done)
        )

        loss = train_step()
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
        'reward': ep_reward,
        'loss': np.mean(ep_loss) if ep_loss else 0,
        'epsilon': epsilon,
        'reached': reason == 'reached',
    })

    if episode % 500 == 0:
        recent    = log[-100:] if len(log) >= 100 else log
        avg_reach = np.mean([r['reached'] for r in recent])
        avg_loss  = np.mean([r['loss'] for r in recent])
        print(f"  Ep {episode:5d}/{N_EPISODES} | ε={epsilon:.3f} | loss={avg_loss:.4f} | reach={avg_reach:.1%}")

torch.save(q_net.state_dict(), 'morl/dqn_model.pt')
df_log = pd.DataFrame(log)
df_log.to_csv('morl/dqn_training_log.csv', index=False)

print()
print("Training finished")
final_reach = df_log['reached'].tail(500).mean()
print(f" Final reach rate: {final_reach:.1%}")
print(f" Model - morl/dqn_model.pt")


#evaluation
print("\nEvaluating RL agent vs Dijkstra baseline")

def run_rl_route(origin, destination):
    state = env.reset(origin=origin, destination=destination)
    total_co2 = 0.0
    total_time = 0.0
    path = [origin]

    for _ in range(MAX_STEPS):
        act_feat, neighbors = env.get_action_features()
        n_valid = len(neighbors)
        if n_valid == 0:
            break

        state_t = torch.tensor(state).unsqueeze(0).to(DEVICE)
        act_feat_t = torch.tensor(act_feat).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            q = q_net(state_t, act_feat_t).squeeze(0)
            mask = torch.full((MAX_ACTIONS,), float('-inf')).to(DEVICE)
            mask[:n_valid] = 0
            action = (q + mask).argmax().item()

        next_node = neighbors[action]
        ed = G.get_edge_data(env.current, next_node)
        if ed is None:
            break
        best = ed[0] if 0 in ed else ed
        total_co2  += float(best.get(CARBON_ATTR, 0))
        total_time += float(best.get(TIME_ATTR,   0))
        path.append(next_node)

        state, _, _, done, reason = env.step(action)
        if done:
            break

    reached = (env.current == destination)
    return total_co2, total_time, reached, path


def run_dijkstra_route(origin, destination):
    def weight_fn(u, v, data):
        best = data[0] if 0 in data else data
        return float(best.get(CARBON_ATTR, 0))
    try:
        path = nx.dijkstra_path(G, origin, destination, weight=weight_fn)
        total_co2 = 0.0
        total_time = 0.0
        for u, v in zip(path[:-1], path[1:]):
            ed = G.get_edge_data(u, v)
            best = ed[0] if 0 in ed else ed
            total_co2  += float(best.get(CARBON_ATTR, 0))
            total_time += float(best.get(TIME_ATTR,   0))
        return total_co2, total_time, True, path
    except nx.NetworkXNoPath:
        return 0, 0, False, []


def run_fastest_route(origin, destination):
    def weight_fn(u, v, data):
        best = data[0] if 0 in data else data
        return float(best.get(TIME_ATTR, 0))
    try:
        path = nx.dijkstra_path(G, origin, destination, weight=weight_fn)
        total_co2 = 0.0
        total_time = 0.0
        for u, v in zip(path[:-1], path[1:]):
            ed = G.get_edge_data(u, v)
            best = ed[0] if 0 in ed else ed
            total_co2  += float(best.get(CARBON_ATTR, 0))
            total_time += float(best.get(TIME_ATTR,   0))
        return total_co2, total_time, True, path
    except nx.NetworkXNoPath:
        return 0, 0, False, []


#Compare among 100 O-D pairs
results = []
sample_pairs = random.sample(reachable_pairs, min(100, len(reachable_pairs)))

for o, d in sample_pairs:
    rl_co2,   rl_time,   rl_reached,   _ = run_rl_route(o, d)
    dijk_co2, dijk_time, dijk_reached, _ = run_dijkstra_route(o, d)
    fast_co2, fast_time, fast_reached, _ = run_fastest_route(o, d)

    if rl_reached and dijk_reached and fast_reached:
        results.append({
            'rl_co2': rl_co2,
            'rl_time': rl_time / 60,
            'dijk_co2': dijk_co2,
            'dijk_time': dijk_time / 60,
            'fast_co2':  fast_co2,
            'fast_time': fast_time / 60,
        })

df_results = pd.DataFrame(results)
print(f"  Evaluated {len(df_results)} complete O-D pairs")

if len(df_results) > 0:
    print(f"\nRoute Comparison")
    print(f" {'Method':<20} {'Avg CO2 (g)':>12} {'Avg Time (min)':>15}")
    print(f"  {'─'*50}")
    print(f" {'Fastest Path':<20} {df_results['fast_co2'].mean():>12.1f} {df_results['fast_time'].mean():>15.2f}")
    print(f"{'RL Agent':<20} {df_results['rl_co2'].mean():>12.1f} {df_results['rl_time'].mean():>15.2f}")
    print(f" {'Dijkstra (carbon)':<20} {df_results['dijk_co2'].mean():>12.1f} {df_results['dijk_time'].mean():>15.2f}")

    co2_savings_rl   = (df_results['fast_co2'] - df_results['rl_co2']).mean()
    co2_savings_dijk = (df_results['fast_co2'] - df_results['dijk_co2']).mean()
    pct_savings_rl   = (co2_savings_rl / df_results['fast_co2'].mean()) * 100
    pct_savings_dijk = (co2_savings_dijk / df_results['fast_co2'].mean()) * 100

    print(f"\n  CO2 savings vs fastest path:")
    print(f"    RL Agent:          {co2_savings_rl:.1f}g  ({pct_savings_rl:.1f}%)")
    print(f"    Dijkstra (carbon): {co2_savings_dijk:.1f}g  ({pct_savings_dijk:.1f}%)")

df_results.to_csv('morl/dqn_route_comparison.csv', index=False)

# Training curves
fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor='k')
for ax in axes:
    ax.set_facecolor('#111111')

window = 100
df_log['reward_smooth'] = df_log['reward'].rolling(window).mean()
df_log['loss_smooth']   = df_log['loss'].rolling(window).mean()
df_log['reach_smooth']  = df_log['reached'].astype(float).rolling(window).mean()

axes[0].plot(df_log['episode'], df_log['reward_smooth'], color='#00c8ff', linewidth=1.5)
axes[0].set_title('Reward (smoothed)', color='white')
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
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')

plt.suptitle('DQN Training Curves (50-node Subgraph)', color='white', fontsize=14)
plt.tight_layout()
plt.savefig('morl/dqn_curves.png', dpi=200, bbox_inches='tight', facecolor='k')
plt.show()
print("Training curves- morl/dqn_curves.png")


#Route comparison scatterplot
if len(df_results) > 0:
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='k')
    ax.set_facecolor('k')

    ax.scatter(df_results['fast_time'], df_results['fast_co2'],
               color='#ff4444', s=60, alpha=0.6, label='Fastest Path', zorder=3)
    ax.scatter(df_results['rl_time'], df_results['rl_co2'],
               color='#00c8ff', s=60, alpha=0.6, label='RL Agent', zorder=4)
    ax.scatter(df_results['dijk_time'], df_results['dijk_co2'],
               color='#00ff88', s=60, alpha=0.6, label='Dijkstra (carbon-optimal)', zorder=5)

    ax.set_xlabel('Travel Time (minutes)', color='white', fontsize=12)
    ax.set_ylabel('CO₂ Emissions (g)', color='white', fontsize=12)
    ax.set_title('Route Comparison: RL Agent vs Baselines\n(50-node DC Subgraph, Heavy Truck Rush Hour)',
                 color='white', fontsize=13)
    ax.tick_params(colors='white')
    ax.legend(facecolor='#222222', labelcolor='white', fontsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    plt.tight_layout()
    plt.savefig('morl/dqn_route_compare.png', dpi=200, bbox_inches='tight', facecolor='k')
    plt.show()
    print("Route comparison- morl/dqn_route_compare.png")

print("\n Final Summary")
print(f"  Device: {DEVICE}")
print(f"  Final reach: {df_log['reached'].tail(500).mean():.1%}")
print(f"  O-D pairs eval: {len(df_results)}")
if len(df_results) > 0:
    print(f"  RL CO2 savings:{pct_savings_rl:.1f}% vs fastest path")
print(f"  Outputs in: morl/")
