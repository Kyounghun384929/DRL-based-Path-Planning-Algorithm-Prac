import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.net(x)

class MAPPOAgent:
    def __init__(self, n_agents, state_dim, action_dim, device):
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Hyperparameters
        self.lr_actor = 3e-5
        self.lr_critic = 1e-4
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.eps_clip = 0.2
        self.k_epochs = 10
        self.entropy_coef = 0.01
        
        # Input dims
        # Actor: sees local state (2) -> Normalized
        # Critic: sees Global state (N*2) + Agent ID (N) -> Normalized
        self.critic_input_dim = (state_dim * n_agents) + n_agents
        
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(self.critic_input_dim).to(device)
        
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': self.lr_actor},
            {'params': self.critic.parameters(), 'lr': self.lr_critic}
        ])
        
        # Memory
        self.buffer = []
        
    def get_action(self, state):
        """
        state: (N, 2) tensor
        """
        with torch.no_grad():
            # Normalize Observation (0~100 -> 0~1)
            norm_state = state / 100.0
            
            probs = self.actor(norm_state)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # Prepare Critic Input
            # Global State: Flatten all agent states
            global_state = norm_state.view(-1) # (N*2)
            global_state_repeat = global_state.unsqueeze(0).expand(self.n_agents, -1) # (N, N*2)
            
            # Agent IDs (One-hot)
            agent_ids = torch.eye(self.n_agents, device=self.device) # (N, N)
            
            # Concat for Citic: [Global State, Agent ID]
            critic_input = torch.cat([global_state_repeat, agent_ids], dim=1) # (N, N*2 + N)
            
            value = self.critic(critic_input).squeeze(-1) # (N,)
            
        return action, log_prob, value
    
    def store_transition(self, transition):
        self.buffer.append(transition)
        
    def update(self):
        if len(self.buffer) == 0:
            return 0.0
            
        # Extract data from buffer
        # List of tuples -> Tuple of lists
        s, a, r, next_s, old_log_p, val, done = map(lambda x: torch.stack(x).to(self.device), zip(*self.buffer))
        
        # Dimensions: (T, N, ...)
        T, N, _ = s.shape
        
        # Ensure shapes are (T, N) to avoid broadcasting errors with values (T, N)
        r = r.view(T, N)
        done = done.view(T, N)
        
        # 1. Normalize States
        s_norm = s / 100.0
        next_s_norm = next_s / 100.0
        
        # 2. Prepare Critic Inputs (Batch processing)
        # s_norm: (T, N, 2) -> Global: (T, N*2)
        global_s = s_norm.view(T, -1) 
        global_next_s = next_s_norm.view(T, -1)
        
        # Expand for each agent: (T, N, N*2)
        global_s_exp = global_s.unsqueeze(1).expand(-1, N, -1)
        global_next_s_exp = global_next_s.unsqueeze(1).expand(-1, N, -1)
        
        # Agent IDs: (N, N) -> (T, N, N)
        eye = torch.eye(N, device=self.device)
        ids = eye.unsqueeze(0).expand(T, -1, -1)
        
        # Final Critic Inputs
        # (T, N, N*2 + N) -> Flatten to (T*N, Input_dim)
        critic_obs = torch.cat([global_s_exp, ids], dim=2).view(-1, self.critic_input_dim)
        critic_next_obs = torch.cat([global_next_s_exp, ids], dim=2).view(-1, self.critic_input_dim)
        
        targets = r.view(-1, 1).float() # (T, N, 1) -> use r (T,N) directly? r is (T,N)
        dones = done.view(-1, 1).float() # (T, N) usually
        
        # 3. Calculate Returns & Advantages (GAE)
        with torch.no_grad():
            values = val.view(T, N)
            next_values = self.critic(critic_next_obs).view(T, N)
            
            advantages = torch.zeros_like(values)
            last_gae_lam = 0
            
            for t in reversed(range(T)):
                if t == T - 1:
                    next_non_terminal = 1.0 - done[t].float()
                    next_val = next_values[t]
                else:
                    next_non_terminal = 1.0 - done[t].float()
                    next_val = values[t+1]
                
                delta = r[t] + self.gamma * next_val * next_non_terminal - values[t]
                last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
                advantages[t] = last_gae_lam
            
            returns = advantages + values

        # 4. Flatten for PPO Update
        # (T, N, ...) -> (T*N, ...)
        b_obs = s_norm.view(-1, self.state_dim)
        b_act = a.view(-1)
        b_log_prob = old_log_p.view(-1)
        b_advantage = advantages.view(-1)
        b_return = returns.view(-1)
        b_critic_obs = critic_obs # Already flattened
        
        # Normalize Advantages
        b_advantage = (b_advantage - b_advantage.mean()) / (b_advantage.std() + 1e-8)
        
        # 5. Optimization Loop
        for _ in range(self.k_epochs):
            # Get current policy outputs
            probs = self.actor(b_obs)
            dist = Categorical(probs)
            new_log_prob = dist.log_prob(b_act)
            entropy = dist.entropy().mean()
            
            # Get current value outputs
            new_values = self.critic(b_critic_obs).squeeze(-1)
            
            # Ratio
            ratio = torch.exp(new_log_prob - b_log_prob)
            
            # Surrogate Loss (Actor)
            surr1 = ratio * b_advantage
            surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * b_advantage
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value Loss (Critic)
            critic_loss = F.mse_loss(new_values, b_return)
            
            # Total Loss
            loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.optimizer.step()
            
        self.buffer = []
        return loss.item()

if __name__ == "__main__":
    from src.envs import Env2DMA
    import random
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = Env2DMA(num_agents=3, device=device, max_episode_steps=500)
    
    agent = MAPPOAgent(
        n_agents=env.num_agents, 
        state_dim=env.state_dim, 
        action_dim=env.action_dim, 
        device=device
    )
    
    episodes = 2000
    
    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0
        done = False
        
        while not done:
            action, log_prob, val = agent.get_action(state)
            
            next_state, reward, dones = env.step(action)
            
            # Store data
            # state, action, reward, next_state, log_prob, value, done
            agent.store_transition((
                state.clone(),
                action,
                reward,
                next_state.clone(),
                log_prob,
                val,
                dones # bool tensor
            ))
            
            state = next_state
            ep_reward += reward.sum().item()
            
            if dones.all():
                done = True
                
        # Update after each episode
        loss = agent.update()
        
        print(f"\n{'='*20} Episode {ep+1}/{episodes} {'='*20}")
        print(f"Total Reward: {ep_reward:.2f} | Loss: {loss:.4f}")
        print("Last Positions:")
        for i in range(env.num_agents):
            pos = state[i].detach().cpu().numpy()
            gate_dist = torch.norm(state[i] - env.goal_pos[i]).item()
            reached = "Reached Goal" if gate_dist < 3.0 else "Not Reached"
            print(f"  - Agent {i}: {np.round(pos, 2)} (Dist: {gate_dist:.2f}) | {reached}")
            
    # Final Test
    print("\nTraining Finished. Testing...")
    state = env.reset()
    print("Initial Positions:", state.cpu().numpy())
    for _ in range(200):
        action, _, _ = agent.get_action(state)
        state, _, dones = env.step(action)
        if dones.all():
            break
        
    torch.save(agent.actor.state_dict(), "mappo_actor_500steps.pth")
    torch.save(agent.critic.state_dict(), "mappo_critic_500steps.pth")
    
    print("Final Positions:")
    for i in range(env.num_agents):
        pos = state[i].cpu().numpy()
        dist = np.linalg.norm(pos - env.goal_pos[i].cpu().numpy())
        dist_tensor = torch.norm(state[i] - env.goal_pos[i]).item()
        reached = "Reached" if dist_tensor < 3.0 else "Not Reached"
        print(f"Agent {i}: {pos} (Dist: {dist_tensor:.2f}) | {reached}")
