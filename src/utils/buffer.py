import torch
import random
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


class PPOBuffer:
    def __init__(self, capacity=100000):
        self.clear()
    
    
    def clear(self):
        self.states    = []
        self.actions   = []
        self.rewards   = []
        self.dones     = []
        self.log_probs = []
        self.values    = []
    
    
    def push(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    
    def compute_gae(self, next_value, gamma=0.99, gae_lambda=0.95):
        batch_size = len(self.rewards)
        advantages = torch.zeros(batch_size, dtype=torch.float32)
        last_gae_lambda = 0
        
        for t in reversed(range(batch_size)):
            if t == batch_size - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_val = self.values[t + 1]
            
            delta = self.rewards[t] + gamma * next_val * next_non_terminal - self.values[t]
            advantages[t] = last_gae_lambda = delta + gamma * gae_lambda * next_non_terminal * last_gae_lambda
        
        returns = advantages + torch.tensor(self.values, dtype=torch.float32)
        return advantages


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

if __name__ == "__main__":
    buffer = ReplayBuffer(capacity=1000)
    buffer.push([1,2,3], 0, 1.0, [1,2,4], False)
    buffer.push([4,5,6], 1, 0.5, [4,5,7], True)
    
    state, action, reward, next_state, done = buffer.sample(2)
    print("States:", state)
    print("Actions:", action)
    print("Rewards:", reward)
    print("Next States:", next_state)
    print("Dones:", done)