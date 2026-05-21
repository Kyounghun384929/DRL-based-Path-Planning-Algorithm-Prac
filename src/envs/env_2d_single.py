# 기존의 Discrete 환경과 Continuous 환경을 통합한 2D 단일 에이전트 환경 구현
# 추가 개선사항으로 목표 중간에 (10x10) 크기의 장애물 추가

from numpy import isin
import torch


class UAVEnv2D:
    def __init__(self, env_type:str, device:str="cuda" if torch.cuda.is_available() else "cpu", **kwargs):
        """
        Args:
        env_type (str): ("discrete" or "continuous")
        device (str): "cuda" or "cpu", defualt "cuda" if available else "cpu"
        state_dim (int): Dimension of the state space (default: 4).
        action_dim (int): Dimension of the action space (discrete -> 4, continuous -> 2).
        max_episode_steps (int): Maximum steps per episode (default: 500).
        """
        self.device            = device
        self.state_dim         = kwargs.get("state_dim", 4)
        self.max_episode_steps = kwargs.get("max_episode_steps", 500)
        self.current_step      = 0
        self.env_type          = env_type.lower()
        
        if env_type == "discrete":
            self.action_dim = kwargs.get("action_dim", 4)
            self.max_speed  = torch.tensor(1.0, device=self.device)
        elif env_type == "continuous":
            self.action_dim = kwargs.get("action_dim", 2)
            self.max_speed  = torch.tensor(1.0, device=self.device)
        else:
            raise ValueError("type must be 'discrete' or 'continuous'")
        
        # --- Initialize environment --- #
        self.env_size = torch.tensor([100.0, 100.0], device=self.device)
        self.reset()
        
        
    def reset(self):
        self.current_step = 0
        self.init_pos = torch.tensor([10.0, 10.0], device=self.device)
        self.goal_pos = torch.tensor([90.0, 90.0], device=self.device)
        self.obs_pos  = torch.tensor([50.0, 50.0], device=self.device)
        self.rel_pos  = self.goal_pos - self.init_pos
        self.state    = torch.cat([self.init_pos, self.rel_pos], dim=0)
        return self.state.clone()
    
    
    def action_space(self, action):
        if self.env_type == "discrete":
            """Grid action space: up, down, left, right"""
            deltas = torch.zeros((2), device=self.device)
            deltas[action == 0] = torch.tensor([0.0, 1.0], device=self.device)
            deltas[action == 1] = torch.tensor([0.0, -1.0], device=self.device)
            deltas[action == 2] = torch.tensor([-1.0, 0.0], device=self.device)
            deltas[action == 3] = torch.tensor([1.0, 0.0], device=self.device)
            return deltas
        
        elif self.env_type == "continuous":
            return action * self.max_speed
    
    
    def step(self, action):
        # Action type check
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, device=self.device, dtype=torch.float32)
        self.current_step += 1
        action_delta = self.action_space(action)
        # self.state += action_delta
        self.state[:2] += action_delta
        self.rel_pos = self.goal_pos - self.state[:2]
        self.state[2:] = self.rel_pos
        reward, done = self.compute_reward_done()
        return self.state.clone(), reward, done
    
    
    def compute_reward_done(self):
        dist_reward_norm = torch.abs(torch.norm(torch.tensor([0.0, 0.0], device=self.device) - self.env_size))
        distance_to_goal = torch.norm(self.state[:2] - self.goal_pos)
        distance_to_obs = torch.norm(self.state[:2] - self.obs_pos)
        
        # done은 Bool 이지만 알고리즘 memory 계산 등을 위해 float32로 저장
        done = torch.tensor(False, device=self.device, dtype=torch.float32)
        if distance_to_goal < 3.0:
            reward = torch.tensor(100.0, device=self.device)
            done = torch.tensor(True, device=self.device, dtype=torch.float32)
        elif distance_to_obs < 15.0:
            reward = torch.tensor(-100.0, device=self.device)
            done = torch.tensor(True, device=self.device, dtype=torch.float32)
        else:
            reward = -distance_to_goal / dist_reward_norm
            done = torch.tensor(False, device=self.device, dtype=torch.float32)
        
        # Boundary Condition
        if torch.any(self.state[:2] <= 0.0) or torch.any(self.state[:2] >= self.env_size):
            reward += torch.tensor(-50.0, device=self.device)
            self.state[:2] = torch.clamp(self.state[:2], torch.tensor([0.0, 0.0], device=self.device), self.env_size)
        
        if self.current_step >= self.max_episode_steps:
            done = torch.tensor(True, device=self.device, dtype=torch.float32)
        
        return reward, done
    
    
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    env_type = "discrete"
    env = UAVEnv2D(env_type=env_type, device=device)
    print(env.obs_pos)
    state = env.reset()
    done = False
    
    total_reward = 0.0
    
    while True:
        action = torch.randint(0, env.action_dim, (1,)).item() if env_type == "discrete" else torch.rand(env.action_dim)
        state, reward, done = env.step(action)
        total_reward += reward.item()
        
        print(f"Step: {env.current_step}, State: {state.cpu().numpy()}, Action: {action}, Reward: {reward.item():.4f}, Done: {done.item()}")
        
        if done.item():
            break
    