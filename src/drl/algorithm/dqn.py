import torch
import torch.optim as optim

from src.drl.network import QNet
from src.utils import ReplayBuffer


class DQNAgent:
    def __init__(
        self, 
        device='cuda' if torch.cuda.is_available() else 'cpu',
        state_dim=None,
        action_dim=None,
        **kwargs
        ):
        self.device = device
        
        # --- State and Action Deimensions (Default: Discrete) --- #
        self.state_dim = state_dim if state_dim is not None else 4
        self.action_dim = action_dim if action_dim is not None else 2
        
        # --- Hyperparameters --- #
        self.lr         = kwargs.get('lr', 1e-4)
        self.gamma      = kwargs.get('gamma', 0.99)
        self.batch_size = kwargs.get('batch_size', 64)
        
        # --- Q Networks --- #
        self.q_net      = QNet(self.state_dim, self.action_dim).to(self.device)
        self.target_net = QNet(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        
        # --- Optimizer --- #
        self.optimizer = optim.AdamW(self.q_net.parameters(), lr=self.lr)
        self.criterion = torch.nn.SmoothL1Loss()
        
        # --- Replay Buffer --- #
        self.replay_buffer = ReplayBuffer(capacity=100000)
        
        # --- Epsilon Decay for Exploration --- #
        self.epsilon = kwargs.get('epsilon', 1.0)
        self.epsilon_decay = kwargs.get('epsilon_decay', 0.995)
        self.epsilon_min = kwargs.get('epsilon_min', 0.0001)
        
        
    def get_action(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        if torch.rand(1).item() < self.epsilon:
            return torch.randint(0, self.action_dim, (1,)).item()
        else:
            with torch.no_grad():
                q_values = self.q_net(state)
            return torch.argmax(q_values, dim=-1).item()
    
    
    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size, self.device)

        states      = torch.stack(states).to(self.device)
        actions     = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards     = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones       = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        curr_q = self.q_net(states).gather(1, actions)
        
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
        
        loss = self.criterion(curr_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
    
    
    def save_model(self, path):
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.q_net.state_dict(), path + 'dqn_qnet.pth')
        torch.save(self.target_net.state_dict(), path + 'dqn_tnet.pth')


if __name__ == "__main__":
    from src.envs import UAVEnv2D
    
    env_type = 'discrete'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    env = UAVEnv2D(env_type, device=device, max_episode_steps=500)
    
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    agent = DQNAgent(device=device, state_dim=state_dim, action_dim=action_dim)
    
    num_episodes = 1000
    target_update_freq = 10
    
    for episode in range(num_episodes):
        state = env.reset()
        state /= 100.0
        episode_reward = 0
        
        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            state /= 100.0
            episode_reward += reward
            
            loss = agent.update()
            
            if done:
                break
        
        agent.update_epsilon()
        
        if episode % target_update_freq == 0:
            agent.update_target_network()
        
        if reward.item() == 100.0:
            reach = True
        else:
            reach = False
        
        print(f"Episode {episode+1}, Reward: {episode_reward:.3f}, Epsilon: {agent.epsilon:.4f}, Last Position: {state.cpu().numpy()}, Reach: {reach}")
    
    print("Training completed.")

    agent.save_model('./db/checkpoints/2D/DQN/')