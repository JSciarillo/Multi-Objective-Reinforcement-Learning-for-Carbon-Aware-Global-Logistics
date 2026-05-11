import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from rl_env import CarbonRoutingEnv
import os

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, is_terminated, mask, next_mask):
        self.buffer.append((state, action, reward, next_state, is_terminated, mask, next_mask))

    def sample(self, batch_size):
        state, action, reward, next_state, is_terminated, mask, next_mask = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), is_terminated, np.array(mask), np.array(next_mask)

    def __len__(self):
        return len(self.buffer)

def train_dqn(graph_path="../data/dc_subgraph_carbon.graphml", episodes=500, batch_size=64, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
    env = CarbonRoutingEnv(graph_path=graph_path, alpha=1.0, beta=1.0, max_steps=100)
    
    state_size = env.observation_space["state"].shape[0]
    action_size = env.action_space.n
    
    q_network = QNetwork(state_size, action_size)
    target_network = QNetwork(state_size, action_size)
    target_network.load_state_dict(q_network.state_dict())
    
    optimizer = optim.Adam(q_network.parameters(), lr=1e-3)
    replay_buffer = ReplayBuffer(10000)
    
    epsilon = epsilon_start
    
    print(f"Training DQN on {graph_path} for {episodes} episodes...")
    
    for episode in range(episodes):
        obs, info = env.reset()
        state = obs["state"]
        mask = obs["action_mask"]
        
        total_reward = 0
        is_terminated = False
        
        while not is_terminated:
            # Epsilon-greedy action selection with masking
            if random.random() < epsilon:
                valid_actions = np.where(mask == 1)[0]
                action = random.choice(valid_actions) if len(valid_actions) > 0 else 0
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    q_values = q_network(state_tensor).squeeze(0).numpy()
                
                # Apply mask (set Q-values of invalid actions to very negative)
                q_values[mask == 0] = -np.inf
                
                if np.all(q_values == -np.inf):
                    action = 0
                else:
                    action = np.argmax(q_values)
                
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = next_obs["state"]
            next_mask = next_obs["action_mask"]
            is_terminated = terminated or truncated
            
            replay_buffer.push(state, action, reward, next_state, is_terminated, mask, next_mask)
            
            state = next_state
            mask = next_mask
            total_reward += reward
            
            if len(replay_buffer) > batch_size:
                states, actions, rewards, next_states, terminations, masks, next_masks = replay_buffer.sample(batch_size)
                
                states_t = torch.FloatTensor(states)
                actions_t = torch.LongTensor(actions).unsqueeze(1)
                rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
                next_states_t = torch.FloatTensor(next_states)
                terminations_t = torch.FloatTensor(terminations).unsqueeze(1)
                next_masks_t = torch.BoolTensor(next_masks)
                
                # Current Q values
                q_values = q_network(states_t).gather(1, actions_t)
                
                # Next Q values from target network
                with torch.no_grad():
                    next_q_values_raw = target_network(next_states_t)
                    # Mask invalid actions
                    next_q_values_raw[~next_masks_t] = -float('inf')
                    # Max Q value over valid actions
                    next_q_values = next_q_values_raw.max(1)[0].unsqueeze(1)
                    # Handle cases where all actions might be invalid (shouldn't happen with proper graph, but safe to check)
                    next_q_values[next_q_values == -float('inf')] = 0.0
                    
                expected_q_values = rewards_t + (1 - terminations_t) * gamma * next_q_values
                
                loss = nn.MSELoss()(q_values, expected_q_values)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        # Update target network
        if episode % 10 == 0:
            target_network.load_state_dict(q_network.state_dict())
            
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        if episode % 50 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")
            
    print("Training finished.")
    os.makedirs("../models", exist_ok=True)
    torch.save(q_network.state_dict(), "../models/dqn_carbon_routing.pth")
    print("Model saved to ../models/dqn_carbon_routing.pth")

if __name__ == "__main__":
    train_dqn(episodes=10000) # Give it 500 episodes to train a bit better