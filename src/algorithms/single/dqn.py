import torch
import torch.nn as nn
import torch.nn.functional as F

import random

from collections import deque
from src.networks.q import QNetwork
from src.networks.noisy_q import QNetwork as NoisyQNetwork


"""
DQN with Noisy Networks for Exploration
"""

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # 튜플 형태로 저장
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, **kwargs):
        self.state_dim     = kwargs.get('state_dim', 2)
        self.action_dim    = kwargs.get('action_dim', 4)
        
        self.lr            = kwargs.get('lr', 0.001)
        self.gamma         = kwargs.get('gamma', 0.99)
        self.device        = kwargs.get('device', 'cpu')
        self.noisy         = kwargs.get('use_noisy', False)
        
        if self.noisy:
            self.q_net = NoisyQNetwork(self.state_dim, self.action_dim).to(self.device)
            self.t_net = NoisyQNetwork(self.state_dim, self.action_dim).to(self.device)
            self.epsilon = 0.0
        else:
            self.q_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
            self.t_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
            self.epsilon       = kwargs.get('epsilon', 1.0)
            self.epsilon_decay = kwargs.get('epsilon_decay', 0.995)
            self.epsilon_min   = kwargs.get('epsilon_min', 0.0001)
        
        self.batch_size = 64
        
        self.t_net.load_state_dict(self.q_net.state_dict())
        self.t_net.eval()
        
        self.optimizer = torch.optim.AdamW(self.q_net.parameters(), lr=self.lr)
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.criterion = nn.SmoothL1Loss()
        
    def get_action(self, state):
        # 텐서 변환 및 배치 차원 추가
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
        state = state.unsqueeze(0).to(self.device) # (1, state_dim)

        # 2. 행동 선택 분기
        if self.noisy:
            self.q_net.reset_noise()
            with torch.no_grad():
                q_values = self.q_net(state)
            return torch.argmax(q_values, dim=1).cpu().item()
            
        else:
            # Epsilon-Greedy
            if torch.rand(1).item() < self.epsilon:
                return torch.randint(0, self.action_dim, (1,)).item()
            else:
                with torch.no_grad():
                    q_values = self.q_net(state)
                return torch.argmax(q_values, dim=1).cpu().item()
    
    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None # Not enough samples to train
        
        # Sampling
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states      = torch.stack([s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32) for s in states]).to(self.device)
        actions     = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards     = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.stack([ns if isinstance(ns, torch.Tensor) else torch.tensor(ns, dtype=torch.float32) for ns in next_states]).to(self.device)
        dones       = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        if self.noisy:
            self.q_net.reset_noise()
            self.t_net.reset_noise()
        
        # Calculate current Q values
        curr_q = self.q_net(states).gather(1, actions)
        
        # Calculate target Q values on next states
        with torch.no_grad():
            max_next_q = self.t_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
        
        # Compute loss
        loss = self.criterion(curr_q, target_q)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_epsilon(self):
        if self.noisy:
            return
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        self.t_net.load_state_dict(self.q_net.state_dict())
    
    
if __name__ == "__main__":
    from src.envs.env_2d import Simple2DGridENV
    
    # 환경 및 에이전트 초기화
    use_noisy = True 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Simple2DGridENV(state_dim=2, action_dim=4, max_episode_steps=200)
    agent = DQNAgent(state_dim=2, action_dim=4, lr=0.0005, use_noisy=use_noisy, device=device)
    
    num_episodes = 500
    target_update_freq = 10 # 10 에피소드마다 타겟 네트워크 동기화

    print(f"Start Training on {agent.device}...")

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # 1. 행동 선택
            action = agent.get_action(state)
            
            # 2. 환경 상호작용
            next_state, reward, done = env.step(action)
            
            # 3. 메모리 저장 (CPU Tensor로 변환하여 저장 권장)
            agent.replay_buffer.push(state.cpu(), action, reward, next_state.cpu(), done)
            
            # 4. 상태 업데이트
            state = next_state
            episode_reward += reward
            
            # 5. 네트워크 학습
            agent.train_step()
            
        # 에피소드 종료 후 처리
        agent.update_epsilon()
        
        if episode % target_update_freq == 0:
            agent.update_target_network()
        
        if episode % 10 == 0:
            print(f"Ep: {episode}, Reward: {episode_reward:.4f}, Epsilon: {agent.epsilon:.2f}")

    print("Training Finished.")