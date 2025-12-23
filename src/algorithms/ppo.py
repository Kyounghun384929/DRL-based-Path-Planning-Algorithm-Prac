import torch
import torch.nn as nn
import torch.optim as optim

class PPOAgent:
    def __init__(self, model, **kwargs):
        self.gamma       = kwargs.get('gamma', 0.99)
        self.eps_clip    = kwargs.get('eps_clip', 0.2)
        self.K_epochs    = kwargs.get('K_epochs', 5)
        self.device      = kwargs.get('device', 'cpu')
        
        # 네트워크 및 옵티마이저
        self.policy    = model.to(self.device)
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=kwargs.get('lr', 0.0003))
        
        # GAE 파라미터
        self.lmbda    = 0.95
        self.mse_loss = nn.MSELoss()
        
        # 데이터 버퍼 (Trajectory 저장용)
        self.data = []
    
    
    def put_data(self, transition):
        """
        transition: (state, action, reward, next_state, prob_a, done)
        """
        self.data.append(transition)
    
    
    def make_batch(self):
        """
        Convert list into tensors
        """
        
        s_lst, a_lst, r_lst, ns_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        
        for transition in self.data:
            s, a, r, ns, prob_a, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            ns_lst.append(ns)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
        
        # list to tensor
        s       = torch.stack(s_lst).float().to(self.device)
        ns      = torch.stack(ns_lst).float().to(self.device)
        a       = torch.tensor(a_lst).to(self.device)
        r       = torch.tensor(r_lst, dtype=torch.float).to(self.device)
        prob_a  = torch.tensor(prob_a_lst, dtype=torch.float).to(self.device)
        done    = torch.tensor(done_lst, dtype=torch.float).to(self.device)
        
        self.data = []
        return s, a, r, ns, prob_a, done
        
        
    def train_step(self):
        # Get batch
        s, a, r, ns, old_log_prob, done = self.make_batch()
        
        # Calculate Monte Carlo Returns (Discounted Rewards)
        rewards = []
        discounted_reward = 0
        # r is tensor (Batch, 1), done is tensor (Batch, 1)
        r_numpy = r.cpu().numpy().flatten()
        done_numpy = done.cpu().numpy().flatten()
        
        for reward, is_not_done in zip(reversed(r_numpy), reversed(done_numpy)):
            if is_not_done == 0:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward * is_not_done)
            rewards.insert(0, discounted_reward)
            
        # Normalize Returns
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device).unsqueeze(1)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        # Calculate Advantage
        with torch.no_grad():
             values = self.policy.critic(s)
             advantage = rewards - values
        
        # PPO Update
        for _ in range(self.K_epochs):
            log_prob, curr_val, entropy = self.policy.evaluate(s, a.squeeze(1))
            
            log_prob = log_prob.unsqueeze(1)
            curr_val = curr_val.unsqueeze(1)
            
            # Ratio: pi_new / pi_old = exp(log_new - log_old)
            ratio = torch.exp(log_prob - old_log_prob)
            
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            
            # Loss
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.mse_loss(curr_val, rewards)
            entropy_loss = -0.01 * entropy.mean()
            
            loss = actor_loss + 0.5 * critic_loss + entropy_loss
            
            # Gradient Descent
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return loss.item()
    
    
if __name__ == "__main__":
    from src.networks import ActorCritic
    
    state_dim  = 4
    action_dim = 2
    ppo_agent = PPOAgent(ActorCritic(state_dim, action_dim), lr=0.0003, gamma=0.99, eps_clip=0.2, K_epochs=4, device='cpu')
    
    # Dummy data for testing
    for _ in range(10):
        state      = torch.randn(state_dim)
        action     = torch.randint(0, action_dim, (1,)).item()
        reward     = torch.randn(1).item()
        next_state = torch.randn(state_dim)
        prob_a     = -0.693
        done       = False
        
        ppo_agent.put_data((state, action, reward, next_state, prob_a, done))
    
    loss = ppo_agent.train_step()
    print("Training loss:", loss)
    print(f"State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}, Prob_a: {prob_a}, Done: {done}")
