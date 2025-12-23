import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[128, 128]):
        super(ActorCritic, self).__init__()
        
        def build_net(output_dim, final_activation=None):
            layers = []
            prev_dim = state_dim
            
            for size in hidden_dims:
                layers.append(nn.Linear(prev_dim, size))
                layers.append(nn.ReLU())
                prev_dim = size
            
            layers.append(nn.Linear(prev_dim, output_dim))
            
            if final_activation:
                layers.append(final_activation)
            
            return nn.Sequential(*layers)
        
        self.actor  = build_net(action_dim, final_activation=nn.Softmax(dim=-1))
        self.critic = build_net(1)
        
    
    def act(self, state) -> tuple:
        """
        Rollout 
        """
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        with torch.no_grad():
            action_probs = self.actor(state)
            dist         = Categorical(action_probs)
            
            action   = dist.sample()
            log_prob = dist.log_prob(action)
            value    = self.critic(state)
        
        return action.detach(), log_prob.detach(), value.detach()
    
    def evaluate(self, state, action) -> tuple:
        """
        Update 
        """
        
        action_probs = self.actor(state)
        dist         = Categorical(action_probs)
        
        log_prob     = dist.log_prob(action)
        entropy      = dist.entropy()
        value        = self.critic(state)
        
        # Critic 출력 차원 (Batch, 1) -> (Batch,)
        return log_prob, value.squeeze(-1), entropy


if __name__ == "__main__":
    state_dim  = 3
    action_dim = 6
    model = ActorCritic(state_dim, action_dim)
    
    sample_state = torch.randn((2, state_dim))
    
    # Test act
    action, log_prob, value = model.act(sample_state)
    print("Sampled actions:", action)
    print("Log probabilities:", log_prob)
    print("State values:", value)
    
    # Test evaluate
    eval_log_prob, eval_value, entropy = model.evaluate(sample_state, action)
    print("Evaluated log probabilities:", eval_log_prob)
    print("Evaluated state values:", eval_value)
    print("Entropy:", entropy)