"""Value Decomposition Networks (VDN) algorithm."""

import torch
import torch.nn as nn
import torch.optim as optim
import random

from collections import deque
from src.networks import QNetwork


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (torch.stack(state),
                torch.stack(action),
                torch.stack(reward),
                torch.stack(next_state),
                torch.stack(done))
        
    def __len__(self):
        return len(self.buffer)


class VDNAgent:
    def __init__(self, device, state_dim, action_dim, **kwargs):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = kwargs.get('lr', 1e-4)
        self.gamma = kwargs.get('gamma', 0.99)

        # Q-Network & Target Network
        self.q_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.t_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.t_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.AdamW(self.q_net.parameters(), lr=self.lr)
    
    
    def get_action(self, state, epsilon):
        # IQL과 동일
        if random.random() < epsilon:
            return torch.randint(0, self.action_dim, (state.size(0),), device=self.device)
        else:
            with torch.no_grad():
                q_values = self.q_net(state) # (N, Action_Dim)
                return q_values.argmax(dim=1) # (N,)
    
    
    def update(self, replay_buffer=ReplayBuffer(), batch_size=128):
        if len(replay_buffer) < batch_size:
            return
        
        # batch sampling
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        # Tensor Shape Check
        # states: (B, N, State_Dim)
        # actions: (B, N)
        # rewards: (B, N, 1)
        
        # Individual Q-value
        curr_q_out = self.q_net(states) # (B, N, Action_Dim)
        curr_q_i = curr_q_out.gather(2, actions.unsqueeze(-1)).squeeze(-1) # (B, N)
        q_total = curr_q_i.sum(dim=1, keepdim=True) # (B, 1)
        
        # Target Q_total
        with torch.no_grad():
            next_q_out = self.t_net(next_states) # (B, N, Action_Dim)
            
            # Max Q-value (or can try Double DQN)
            max_next_q_i = next_q_out.max(dim=2)[0] # (B, N)
            target_q_total = max_next_q_i.sum(dim=1, keepdim=True) # (B, 1)
            
        
        # calculate global reward
        global_reward = rewards.sum(dim=1)
        
        # Done : if all agents are done -> is done
        global_done = dones.all(dim=1).float() # (B, 1)
        
        # TD Target
        target_q = global_reward + self.gamma * target_q_total * (1 - global_done)
        
        # Update Loss
        loss = nn.MSELoss()(q_total, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        
    def update_target(self):
        self.t_net.load_state_dict(self.q_net.state_dict())
        
        
if __name__ == "__main__":
    from src.envs import Env2DMA

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = Env2DMA(num_agents=3, device=device)
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    agent = VDNAgent(device, state_dim, action_dim)
    replay_buffer = ReplayBuffer()
    
    episodes = 1000
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while True:
            action = agent.get_action(state, epsilon)
            next_state, reward, dones = env.step(action)
            
            replay_buffer.push(state, action, reward, next_state, dones)
            state = next_state
            episode_reward += reward.sum().item()
            
            agent.update(replay_buffer)
            
            # state = next_state
            
            if dones.all():
                break
            
        if episode % 10 == 0:
            agent.update_target()
        
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        
        print(f"\n{'='*20} Episode {episode+1}/{episodes} {'='*20}")
        print(f"Total Reward: {episode_reward:.5f}, Epsilon: {epsilon:.4f}")
        print("Last Positions:")
        for i, pos in enumerate(state.tolist()):
            # Check if reached goal
            goal = env.goal_pos[i]
            dist = torch.norm(state[i] - goal).item()
            reached = "Reached goal" if dist < 3.0 else "Not reached"
            print(f"  - Agent {i}: {[round(x, 2) for x in pos]} | {reached} (Dist: {dist:.2f})")
        print(f"{'='*50}")