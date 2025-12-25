import torch
import torch.nn as nn


class DDPG_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[128, 128]):
        super(DDPG_Actor, self).__init__()
        
        layers = []
        prev_dim = state_dim
        
        for size in hidden_dims:
            layers.append(nn.Linear(prev_dim, size))
            layers.append(nn.ReLU())
            prev_dim = size
        
        self.layers = nn.Sequential(*layers)
        
        # Output layer
        self.out = nn.Linear(prev_dim, action_dim)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, state):
        x = self.layers(state)
        return torch.tanh(self.out(x))  # Assuming action space is normalized between -1 and 1
    
    
class DDPG_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[128, 128]):
        super(DDPG_Critic, self).__init__()
        
        layers = []
        prev_dim = state_dim + action_dim  # Critic takes both state and action as input
        
        for size in hidden_dims:
            layers.append(nn.Linear(prev_dim, size))
            layers.append(nn.ReLU())
            prev_dim = size
        
        self.layers = nn.Sequential(*layers)
        
        # Output layer
        self.out = nn.Linear(prev_dim, 1)  # Q-value output
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, state, action):
        # Concat (Batch, State) + (Batch, Action) -> (Batch, State + Action)
        x = torch.cat([state, action], dim=1)
        x = self.layers(x)
        return self.out(x)  # Q-value output