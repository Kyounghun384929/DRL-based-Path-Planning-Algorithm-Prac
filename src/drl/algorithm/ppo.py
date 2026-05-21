import torch
import torch.nn as nn
import torch.optim as optim

from torch.distributions import Categorical

from src.drl.network import ActorCritic
from src.utils import PPOBuffer, RolloutBuffer

class PPOAgent:
    def __init__(self, device, env_type, state_dim, action_dim, **kwargs):
        self.device = device
        self.env_type = env_type
        
        if self.env_type == 'continuous':
            self.action_std = kwargs.get('action_std', 0.6)
        
        # --- State and Action Dimensions --- #
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # --- Hyperparameters --- #
        self.gamma       = kwargs.get('gamma', 0.99)
        self.eps_clip    = kwargs.get('eps_clip', 0.2)
        self.K_epochs    = kwargs.get('K_epochs', 20)
        self.lr_actor    = kwargs.get('lr_actor', 3e-5)
        self.lr_critic   = kwargs.get('lr_critic', 1e-4)
        
        # --- Buffer (Rollout X, Replay Memory O) --- #
        # self.buffer = PPOBuffer(capacity=10000)
        self.buffer = RolloutBuffer()
        
        # --- Actor and Critic Networks --- #
        self.policy = ActorCritic(self.state_dim, self.action_dim, self.env_type).to(self.device)
        self.optimizer = optim.AdamW([
            {'params': self.policy.actor.parameters(), 'lr': self.lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': self.lr_critic}
        ])
        self.policy_old = ActorCritic(self.state_dim, self.action_dim, self.env_type).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.mse_loss = nn.MSELoss()
        
        
    def set_action_std(self, new_action_std):
        if self.env_type == 'continuous':
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
    
    
    def decay_action_std(self, action_std_decay_rate, min_action_std):
        if self.env_type == 'continuous':
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = torch.round(self.action_std, decimals=4)
            
            if self.action_std <= min_action_std:
                self.action_std = min_action_std
            
            self.set_action_std(self.action_std)
    
    
    def get_action(self, state):
        with torch.no_grad():
            action, action_logprob, state_val = self.policy_old.act(state)
    
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)
        
        if self.env_type == 'continuous':
            return torch.clamp(action, -1.0, 1.0).detach().cpu().numpy().flatten()
        else:
            return action.item()
    
    
    def update(self, next_state, is_truncated):
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)

        with torch.no_grad():
            if is_truncated:
                next_val_boundary = self.policy.critic(next_state).detach()
            else:
                next_val_boundary = torch.tensor([0.0], device=self.device)

        rewards = []
        gae = 0
        gae_lambda = 0.95
        
        for i in reversed(range(len(self.buffer.rewards))):
            if i == len(self.buffer.rewards) - 1:
                next_val = next_val_boundary
            else:
                next_val = old_state_values[i + 1]
            
            if self.buffer.is_terminals[i] and not is_truncated:
                next_val = 0
                gae = 0

            delta = self.buffer.rewards[i] + self.gamma * next_val - old_state_values[i]
            gae = delta + self.gamma * gae_lambda * gae
            rewards.insert(0, gae + old_state_values[i])
        
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        
        advantages = rewards - old_state_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            state_values = torch.squeeze(state_values)
            
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(state_values, rewards) - 0.01 * dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()
        
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.buffer.clear()
        
        
    def save_model(self, path):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy_old.state_dict(), path + 'state_norm_ppo.pth')


if __name__ == "__main__":
    from src.envs.env_2d_single import UAVEnv2D
    
    # env_type = 'discrete'
    env_type = 'continuous'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    env = UAVEnv2D(env_type, device, max_episode_steps=1000)
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    agent = PPOAgent(device, env_type, state_dim, action_dim)
    
    num_episodes = 1000
    
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        state /= 100.0
        episode_reward = 0
        
        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            
            # Check for wall collision (to stop accumulating massive negative rewards)
            is_wall_hit = torch.any(next_state[:2] <= 0.0) or torch.any(next_state[:2] >= env.env_size)
            if is_wall_hit:
                reward = torch.tensor(-600.0, device=device)
                done = torch.tensor(True, device=device)
            
            # reward normalization
            reward = reward / 100.0
            
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)
            
            state = next_state
            state /= 100.0
            episode_reward += reward
            
            if done:
                # 500 step timeout is truncation (if not a wall/goal/obstacle hit)
                is_truncated = (env.current_step >= env.max_episode_steps) and not is_wall_hit and (reward.item() < 50.0 and reward.item() > -100.0)
                agent.update(next_state, is_truncated)
                break
        
        if reward.item() >= 0.5: # 50.0 / 100.0
            reach = True
        else:
            reach = False
            
        print(f"Episode {episode} Reward: {episode_reward.item():.2f}, Last Position: [{state[0]:.1f}, {state[1]:.1f}], Step: {env.current_step}, Reach: {reach}")
        
    print("Training finished.")
    agent.save_model('./db/checkpoints/2D/PPO/Continuous/')