import torch

class OUNoise:
    def __init__(self, action_dim, **kwargs):
        self.device     = kwargs.get("device", "cpu")
        self.action_dim = action_dim
        self.mu         = kwargs.get("mu", 0.0)
        self.theta      = kwargs.get("theta", 0.15)
        self.sigma      = kwargs.get("sigma", 0.2)
        self.state      = None
        self.reset()
        
    def reset(self):
        self.state = torch.ones(self.action_dim, device=self.device) * self.mu
    
    def sample(self, mode="Ornstein-Uhlenbeck"):
        """
        mode: "Gaussian" or "Ornstein-Uhlenbeck"
        """
        x = self.state
        if mode == "Gaussian": # To be Fixed
            return torch.normal(self.mu, self.sigma, size=(self.action_dim,), device=self.device)
        else: # Use Ornstein-Uhlenbeck process
            dx = self.theta * (self.mu - x) + self.sigma * torch.randn(len(x), device=self.device)
        self.state = x + dx
        return self.state