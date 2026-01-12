import torch

"""2D Grid Environment for Multi-Agent Reinforcement Learning"""

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Env2DMA:
    def __init__(self, num_agents, device="cpu", **kwargs):
        """
        Args:
            num_agents (int): Number of agents in the environment
            device (str or torch.device): Device to run the environment on
        """
        self.num_agents        = num_agents
        self.device            = device
        
        # Multi-agent environment parameters
        # For batch, each dimension is expanded by num_agents [2] -> [num_agents, 2]
        self.state_dim         = kwargs.get("state_dim", 2) # Agent가 보는 정보의 크기 -> 고정 1차원 vector
        self.action_dim        = kwargs.get("action_dim", 4) # Agnet가 취할 수 있는 행동의 크기 -> 상, 하, 좌, 우
        self.max_episode_steps = kwargs.get("max_episode_steps", 200)
        self.current_step      = 0
        
        self.env_size          = torch.tensor([100.0, 100.0], device=self.device)
        
        # Action tables
        self.action_deltas = torch.tensor([
            [0.0, 1.0],   # Up
            [0.0, -1.0],  # Down
            [-1.0, 0.0],  # Left
            [1.0, 0.0]    # Right
        ], device=self.device)  # [action_dim, 2]
        
        # [Num_agents, 2]
        self.current_step = 0
        self.state = self.reset()
    
    
    def reset(self):
        self.current_step = 0
        
        init_x = self._trans_tensor(5.0)
        init_y = self._trans_tensor(5.0)
        goal_x = self._trans_tensor(95.0)
        goal_y = self._trans_tensor(95.0)
        spacing = self._trans_tensor(2.0)
        
        # Agent들의 위치 생성을 위한 인덱스 생성
        index = torch.arange(self.num_agents, device=self.device, dtype=torch.float32)
        self.state = torch.stack([
            init_x + index * spacing,
            init_y.expand(self.num_agents)
        ], dim=1)  # [num_agents, 2]
        
        self.goal_pos = torch.stack([
            goal_x - index * spacing,
            goal_y.expand(self.num_agents)
        ], dim=1)  # [num_agents, 2]
        
        return self.state.clone()
    
    
    def step(self, actions):
        """
        Args:
            actions (torch.Tensor): Actions taken by agents, shape [num_agents]
        
        Returns:
            next_state (torch.Tensor): Next state of agents, shape [num_agents, state_dim]
            rewards (torch.Tensor): Rewards received by agents, shape [num_agents]
            done (bool): Whether the episode has ended
            info (dict): Additional information
        """
        self.current_step += 1
        
        # Update states based on actions
        deltas = self.action_deltas[actions]  # [num_agents, 2]
        self.state += deltas
        
        self.state = torch.clamp(self.state, torch.tensor(0.0, device=self.device), self.env_size)
        
        # Calculate rewards
        distances = torch.norm(self.state - self.goal_pos, dim=1)
        rewards = -distances / 100.0
        
        # Check for done
        success_mask = distances < 3.0
        timeout = self.current_step >= self.max_episode_steps
        
        done = (success_mask | timeout).unsqueeze(1)  # (N, 1)
        
        rewards[success_mask] = 100.0
        
        return self.state.clone(), rewards.unsqueeze(1), done # (N, 1)
    
    
    
    def _trans_tensor(self, value):
        if not torch.is_tensor(value):
            value = torch.tensor(value, device=self.device, dtype=torch.float32)
        return value
    

if __name__ == "__main__":
    from rich import print
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Env2DMA(num_agents=10, device=device)
    
    state = env.reset()
    print("Initial State:", state)
    
    for step in range(5):
        actions = torch.randint(0, 4, (env.num_agents,), device=device)
        next_state, rewards, dones = env.step(actions)
        print(f"Step {step+1}:")
        print("  Actions:", actions.tolist())
        print("  Next State:", next_state.cpu().numpy())
        print("  Rewards:", rewards.cpu().numpy())
        print("  Dones:", dones.cpu().numpy())
    
