import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical, MultivariateNormal

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, env_type, action_std_init=0.6):
        super(ActorCritic, self).__init__()
        
        self.env_type = env_type
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        if self.env_type == 'continuous':
            self.action_dim = action_dim
            self.register_buffer('action_var', torch.full((self.action_dim,), action_std_init * action_std_init))
            
        if self.env_type == 'continuous':
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, action_dim),
                nn.Tanh()
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, action_dim),
                nn.Softmax(dim=-1)
            )
            
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self._init_weights()
    
    def _init_weights(self):
        def init_linear(layer, gain):
            nn.init.orthogonal_(layer.weight, gain=gain)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

        for layer in [self.actor[0], self.actor[2], self.critic[0], self.critic[2]]:
            init_linear(layer, nn.init.calculate_gain('relu'))
            
        init_linear(self.actor[4], 0.01)
        init_linear(self.critic[4], 1.0)
    
    
    def set_action_std(self, new_action_std):
        if self.env_type == 'continuous':
            self.action_var.fill_(new_action_std * new_action_std)
    
    def forward(self):
        raise NotImplementedError
    
    
    def act(self, state):
        if self.env_type == 'continuous':
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:  # discrete
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_value = self.critic(state)
        
        return action.detach(), action_logprob.detach(), state_value.detach()
    
    
    def evaluate(self, state, action):
        if self.env_type == 'continuous':
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            if self.state_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:  # discrete
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        
        return action_logprobs, state_value, dist_entropy