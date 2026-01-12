import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # Actor network for discrete action space using softmax
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        """
        상태 state를 입력받아, Categorical 분포에서 action을 샘플링하고
        그 log_prob 및 상태 가치(state value)를 반환.
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)  # (1,B)
        
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        state_value = self.critic(state)
        
        return action, action_logprob, state_value

    def evaluate(self, states, actions):
        action_probs = self.actor(states)
        dist = Categorical(action_probs)
        
        action_logprob = dist.log_prob(actions)
        entropy = dist.entropy()
        state_values = self.critic(states).squeeze(-1)  # (B,)

        return action_logprob, state_values, entropy


class Distributional_A2C(nn.Module):
    def __init__(self, state_dim, action_dim, n_quantiles=51):
        super(Distributional_A2C, self).__init__()
        # Actor network for discrete action space using softmax
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        # Critic network
        self.nq = n_quantiles
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, n_quantiles)
        )

        # taus ActorCritic 레벨에 저장(critic 내부 X)
        self.register_buffer(
            "taus",
            (torch.arange(1, n_quantiles + 1, dtype=torch.float32) - 0.5)
            / n_quantiles
        )
    
    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)  # (1,B)
            
        probs  = self.actor(state)
        dist   = Categorical(probs)
        action = dist.sample()
        logp   = dist.log_prob(action)
        
        q = self.critic(state)  # (1, nq)
        v = q.mean()            # 기대값만 정책에 사용
        return action, logp, v
    
    def evaluate(self, states, actions):
        """
        ▸ states  : (B, obs_dim)
        ▸ actions : (B,)      discrete index
        반환 순서 : logp, V_mean, entropy
        """
        probs = self.actor(states)
        dist  = Categorical(probs)
        
        action_logprobs = dist.log_prob(actions)
        dist_entropy   = dist.entropy()
        
        q_values = self.critic(states)  # (B, nq)
        v_values = q_values.mean(dim=1) # (B,)
        
        return action_logprobs, v_values, dist_entropy, q_values


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(), # PPO는 보통 Tanh를 선호
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, x):
        logits = self.net(x)
        return F.softmax(logits, dim=-1)


# Critic: 현재 상태의 가치 평가 (Global State -> Value)
class Critic(nn.Module):
    def __init__(self, global_state_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.net(x)



if __name__ == "__main__":
    state_dim  = 3
    action_dim = 6
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ActorCritic(state_dim, action_dim).to(device)
    
    sample_state = torch.randn((2, state_dim)).to(device)
    
    # Test act
    action, action_logprob, state_value = model.act(sample_state)
    print("Sampled actions:", action)
    print("Log probabilities:", action_logprob)
    print("State values:", state_value)
    
    # Test evaluate
    eval_log_prob, eval_value, entropy = model.evaluate(sample_state, action)
    print("Evaluated log probabilities:", eval_log_prob)
    print("Evaluated state values:", eval_value)
    print("Entropy:", entropy)
    
    # Test Distributional_A2C
    print("")
    dist_model = Distributional_A2C(state_dim, action_dim).to(device)
    action_d, logp_d, v_d = dist_model.act(sample_state)
    print("Distributional Sampled actions:", action_d)
    print("Distributional Log probabilities:", logp_d)
    print("Distributional State values:", v_d)
    
    # Test evaluate for Distributional_A2C
    eval_log_prob_d, eval_value_d, entropy_d, q_values_d = dist_model.evaluate(sample_state, action_d)
    print("Distributional Evaluated log probabilities:", eval_log_prob_d)
    print("Distributional Evaluated state values:", eval_value_d)
    print("Distributional Entropy:", entropy_d)
    