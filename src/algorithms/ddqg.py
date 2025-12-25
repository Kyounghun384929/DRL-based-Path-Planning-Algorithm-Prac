import torch
import torch.nn as nn
import torch.optim as optim

import random

from collections import deque
from src.networks import DDPG_Actor, DDPG_Critic
from src.utils import OUNoise, GaussianNoise, convert_to_tensor


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(torch.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class DDPGAgent:
    def __init__(self, actor, critic, **kwargs):
        self.device      = kwargs.get("device", DEVICE)
        self.gamma       = kwargs.get("gamma", 0.99)
        self.tau         = kwargs.get("tau", 0.005)
        self.batch_size  = kwargs.get("batch_size", 64)
        self.actor_lr    = kwargs.get("actor_lr", 1e-4)
        self.critic_lr   = kwargs.get("critic_lr", 1e-3)
        self.buffer_size = kwargs.get("buffer_size", 1000000)
        
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        
        # Target networks (Copy of the original networks)
        self.target_actor  = actor.to(self.device)
        self.target_critic = critic.to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Optim
        self.actor_optimizer  = optim.AdamW(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=self.critic_lr)
        
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        self.criterion     = nn.MSELoss()
        
    def get_action(self, state, noise=None):
        """
        State -> Actor -> Action + Noise -> Clip
        """
        
        state = convert_to_tensor(state, device=self.device)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state) # (action_dim, )
        self.actor.train()
        
        # Add noise for exploration
        if noise is not None:
            noise_val = noise.sample()
            action += noise_val
        
        
        return action.clamp(-1.0, 1.0)
    
    def soft_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + source_param.data * self.tau
                )
    
    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        
        # Convert to Tensor
        states = torch.stack([s if isinstance(s, torch.Tensor) else convert_to_tensor(s, device=self.device) for s in state]).to(self.device)
        next_states = torch.stack([ns if isinstance(ns, torch.Tensor) else convert_to_tensor(ns, device=self.device) for ns in next_state]).to(self.device)
        
        actions = torch.tensor(action, device=self.device)
        rewards = torch.tensor(reward, device=self.device).unsqueeze(1)
        dones   = torch.tensor(done, device=self.device).unsqueeze(1)
        
        # Critic update
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = self.target_critic(next_states, next_actions)
            
            # y = r + gamma * Q'(s', a') * (1 - done) Bellman
            target_y = rewards + self.gamma * target_q * (1 - dones)
            
        current_q = self.critic(states, actions)
        
        critic_loss = self.criterion(current_q, target_y)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor update
        pred_actions = self.actor(states)
        actor_loss = -self.critic(states, pred_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        self.soft_update(self.target_actor, self.actor)
        self.soft_update(self.target_critic, self.critic)
        
        return actor_loss.item(), critic_loss.item()
    
    
if __name__ == "__main__":
    from src.envs import Simple2DContinuousENV
    env = Simple2DContinuousENV()
    
    state_dim  = env.state_dim
    action_dim = env.action_dim
    
    actor  = DDPG_Actor(state_dim, action_dim)
    critic = DDPG_Critic(state_dim, action_dim)
    
    agent = DDPGAgent(actor, critic)
    
    # Example state
    state = torch.randn(state_dim).to(DEVICE)
    
    # Get action
    action = agent.get_action(state)
    print("Action:", action)