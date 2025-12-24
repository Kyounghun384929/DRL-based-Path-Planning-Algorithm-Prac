import torch
import torch.nn as nn
import torch.optim as optim

from src.networks import ActorCritic, Distributional_A2C

# Set device
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')


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

class PPOAgent:
    def __init__(self, model, **kwargs):
        # 하이퍼파라미터 설정
        self.gamma       = kwargs.get('gamma', 0.99)
        self.eps_clip    = kwargs.get('eps_clip', 0.2)
        self.K_epochs    = kwargs.get('K_epochs', 4)
        
        self.buffer      = RolloutBuffer()
        self.device      = kwargs.get('device', device)
        
        self.policy      = model.to(self.device)
        self.optimizer   = optim.AdamW([
            {'params': self.policy.actor.parameters(), 'lr': kwargs.get('lr_actor', 3e-5)},
            {'params': self.policy.critic.parameters(), 'lr': kwargs.get('lr_critic', 1e-4)}
        ])
        self.policy_old  = model.to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.mse_loss = nn.MSELoss()
        
    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob, state_value = self.policy_old.act(state)
        
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_value)
        return action.item()
    
    def ppo_update(self):
        rewards = []
        discounted_reward = 0
        
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # convert buffer to tensors
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).to(self.device)
        
        advantages = rewards.detach() - old_state_values.detach()
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            
            ratios = torch.exp(logprobs - old_logprobs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(state_values, rewards) - 0.01 * dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
        
    def save(self, filepath):
        torch.save(self.policy.state_dict(), filepath)
        
    def load(self, filepath):
        self.policy_old.load_state_dict(torch.load(filepath, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(filepath, map_location=lambda storage, loc: storage))
    

if __name__ == "__main__":
    from src.envs import get_env
    
    env = get_env("2d")
    state_dim = 2
    action_dim = 4
    
    model = ActorCritic(state_dim, action_dim)
    agent = PPOAgent(model)
    
    state = env.reset()
    for t in range(100):
        action = agent.select_action(state)
        state, reward, done = env.step(action)
        agent.buffer.rewards.append(reward)
        agent.buffer.is_terminals.append(done)
        
        print(f"Step {t}: Action {action}, Reward {reward}, Done {done}")
        
        if done:
            state = env.reset()
        
        if (t + 1) % 20 == 0:
            agent.ppo_update()
    
    print("PPO Agent test completed.")