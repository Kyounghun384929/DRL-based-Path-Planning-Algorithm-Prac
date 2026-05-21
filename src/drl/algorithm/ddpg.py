import torch
import torch.nn as nn
import torch.optim as optim

from src.drl.network import DDPGActor as Actor, DDPGCritic as Critic
from src.utils import ReplayBuffer


class DDPGAgent:
    def __init__(self, device, state_dim, action_dim, **kwargs):
        self.device      = device
        
        # --- Hyperparameters --- #
        self.gamma       = kwargs.get("gamma", 0.99)
        self.tau         = kwargs.get("tau", 0.005)
        self.batch_size  = kwargs.get("batch_size", 64)
        self.actor_lr    = kwargs.get("actor_lr", 1e-4)
        self.critic_lr   = kwargs.get("critic_lr", 5e-4)
        self.buffer_size = kwargs.get("buffer_size", 1000000)
        
        # --- Networks --- #
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        
        # --- Target networks --- #
        self.target_actor  = Actor(state_dim, action_dim).to(self.device)
        self.target_critic = Critic(state_dim, action_dim).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # --- Optimizers --- #
        self.actor_optimizer  = optim.AdamW(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=self.critic_lr)
        
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        self.criterion     = nn.MSELoss()
    
    
    def get_action(self, state):
        '''
        Noise -> Gaussian Noise for Simplicity
        '''
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        with torch.no_grad():
            action = self.actor(state)
        
        noise = torch.randn_like(action) * 0.1
        action = torch.clamp(action + noise, -1.0, 1.0)
        
        return action.squeeze(0)
    
    
    def _soft_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + source_param.data * self.tau
                )
    
    
    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)
        
        states      = torch.stack(state).to(self.device).float()
        actions     = torch.stack(action).to(self.device).float()
        rewards     = torch.stack(reward).to(self.device).float().view(-1, 1)
        next_states = torch.stack(next_state).to(self.device).float()
        dones       = torch.stack(done).to(self.device).float().view(-1, 1)
        
        # --- Update Critic --- #
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q_values = self.target_critic(next_states, next_actions)
            
            y = rewards + self.gamma * target_q_values * (1 - dones)
        
        curr_q = self.critic(states, actions)
        
        critic_loss = self.criterion(curr_q, y)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        ########################
        
        # --- Update Actor --- #
        pred_actions = self.actor(states)
        actor_loss = -self.critic(states, pred_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        ########################
        
        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)
    
    
    def save_model(self, path):
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), path + "ddpg_actor.pth")
        torch.save(self.critic.state_dict(), path + "ddpg_critic.pth")
    
    

if __name__ == "__main__":
    from src.envs import UAVEnv2D
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = UAVEnv2D(env_type='continuous', device=device, max_episode_steps=2000)
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    agent = DDPGAgent(device=device, state_dim=state_dim, action_dim=action_dim)
    
    num_episodes = 1000
    
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        state /= 100.0
        episode_reward = 0.0
        
        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            
            # Check for wall collision (to stop accumulating massive negative rewards)
            is_wall_hit = torch.any(next_state[:2] <= 0.0) or torch.any(next_state[:2] >= env.env_size)
            if is_wall_hit:
                reward = torch.tensor(-600.0, device=device)
                done = torch.tensor(True, device=device)
            
            # reward normalization
            reward /= 100.0
            
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            state /= 100.0
            episode_reward += reward.item()
            
            agent.update()
            
            if done:
                break
        
        if reward.item() >= 0.5:
            reach = True
        else:
            reach = False
        
        print(f"Episode {episode} | Reward: {episode_reward:.2f} | Last Position: [{state[0]:.1f}, {state[1]:.1f}] | Step: {env.current_step} | Reach Goal: {reach}")
    
    print("Training Finished.")
    
    agent.save_model('./db/checkpoints/2D/DDPG/')