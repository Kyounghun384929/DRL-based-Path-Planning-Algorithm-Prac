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
    

...