import torch

class MultiUAVEnv2D:
    def __init__(
            self, 
            num_agents: int, 
            env_type:str, 
            device:str="cuda" if torch.cuda.is_available() else "cpu", 
            **kwargs
            ):
        self.num_agents = num_agents
        self.device     = device
        
        self.state_dim         = kwargs.get("state_dim", 6)
        self.max_episode_steps = kwargs.get("max_episode_steps", 500)
        self.current_step      = 0
        self.env_type          = env_type.lower()
        
        self.max_speed = torch.tensor(1.0, device=self.device)
        self.state_dim = 6
        
        if self.env_type == "discrete":
            self.action_dim = 4
            self.max_speed  = torch.tensor(1.0, device=self.device)
        elif self.env_type == "continuous":
            self.action_dim = 2
            self.max_speed  = torch.tensor(1.0, device=self.device)
        else:
            raise ValueError("type must be 'discrete' or 'continuous'")
        
        # --- Initialize environment --- #
        self.env_size = torch.tensor([100.0, 100.0], device=self.device)
        self.reset()
        
        
    def reset(self):
        self.current_step = 0
        
        # --- Agent and Goal Positions --- #
        corners = torch.tensor([
            [5.0, 5.0],
            [5.0, 95.0],
            [95.0, 95.0],
            [95.0, 5.0]
        ], device=self.device)
        
        agent_indices = torch.arange(self.num_agents, device=self.device)
        
        start_corner_idx = agent_indices % 4
        goal_corner_idx  = (start_corner_idx + 2) % 4
        
        self.state    = corners[start_corner_idx].clone()
        self.goal_pos = corners[goal_corner_idx].clone()
        
        # 장애물 위치는 offset 적용 전 기준 위치의 중간값
        self.obs_pos = (self.state + self.goal_pos) / 2.0

        if self.num_agents > 4:
            # 충돌 방지를 위해 offset을 3.0보다 크게 설정 (예: 4.0)
            offset = (agent_indices // 4).unsqueeze(1).float() * 4.0
            self.state += torch.where(self.state < 50.0, offset, -offset)
        
        self.dones = torch.zeros(self.num_agents, dtype=torch.bool, device=self.device)
        
        return self._get_obs()
    
    def _get_obs(self):
        return torch.cat([self.state, self.goal_pos, self.obs_pos], dim=1)
    
    
    def action_spaces(self, actions):
        if self.env_type == "discrete":
            deltas = torch.zeros((self.num_agents, 2), device=self.device)
            # actions: [num_agents] 또는 [num_agents, 1] 형태의 정수 텐서 가정
            if actions.dim() > 1 and actions.shape[1] > 1:
                actions = torch.argmax(actions, dim=1)
            
            actions_flat = actions.view(-1)
            deltas[actions_flat == 0] = torch.tensor([0.0, 1.0], device=self.device)
            deltas[actions_flat == 1] = torch.tensor([0.0, -1.0], device=self.device)
            deltas[actions_flat == 2] = torch.tensor([-1.0, 0.0], device=self.device)
            deltas[actions_flat == 3] = torch.tensor([1.0, 0.0], device=self.device)
            return deltas
        return actions * self.max_speed
    
    def step(self, actions):
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, device=self.device, dtype=torch.float32)
        
        self.current_step += 1
        
        # 2. 이동 계산 및 적용 (이미 끝난 에이전트는 고정)
        deltas = self.action_spaces(actions)
        active_mask = (~self.dones).float().unsqueeze(1)
        self.state += deltas * active_mask
        self.state = torch.clamp(self.state, 0.0, self.env_size[0])
        
        # 3. 보상 및 종료 조건 계산
        rewards, dones = self.compute_reward_done()
        
        # 상태 업데이트
        self.dones = dones.squeeze(1).bool()
        
        return self._get_obs(), rewards, dones
    
    
    def compute_reward_done(self):
        distances = torch.norm(self.state - self.goal_pos, dim=1)
        
        dist_reward_norm = torch.abs(torch.norm(torch.tensor([0.0, 0.0], device=self.device) - self.env_size))
        
        rewards = -distances / dist_reward_norm
        
        success_mask = distances < 3.0
        
        dist_obs = torch.cdist(self.state, self.obs_pos)
        min_dist_to_obs, _ = torch.min(dist_obs, dim=1)
        collision_mask = min_dist_to_obs < 5.0
        
        # --- Agent-Agent Collision --- #
        collision_agent_mask = torch.zeros(self.num_agents, dtype=torch.bool, device=self.device)
        if self.num_agents > 1:
            dist_agents = torch.cdist(self.state, self.state)
            # 자기 자신과의 거리는 제외 (무한대로 설정)
            dist_agents += torch.eye(self.num_agents, device=self.device) * 1e5
            min_dist_agents, _ = torch.min(dist_agents, dim=1)
            collision_agent_mask = min_dist_agents < 3.0
            
        rewards[success_mask] = 100.0
        rewards[collision_mask] = -100.0
        rewards[collision_agent_mask] = -100.0

        # --- Boundary Penalty --- #
        out_of_bounds = (self.state < 0.0) | (self.state > self.env_size)
        boundary_mask = torch.any(out_of_bounds, dim=1)
        rewards[boundary_mask] -= 50.0
        
        # 이미 이전에 끝난 에이전트는 보상 0
        rewards[self.dones] = 0.0

        current_dones = success_mask | collision_mask | collision_agent_mask
        
        timeout = self.current_step >= self.max_episode_steps
        if timeout:
            current_dones[:] = True
            
        final_dones = current_dones | self.dones
        
        return rewards.unsqueeze(1), final_dones.unsqueeze(1).float()
        

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from kkh_utils import apply_research_style
    apply_research_style()
    
    env = MultiUAVEnv2D(num_agents=4, env_type="discrete")
    state = env.reset()
    done = torch.zeros((env.num_agents, 1), dtype=torch.bool, device=env.device)
    
    total_reward = 0.0
    
    while not torch.all(done):
        if env.env_type == "discrete":
            action = torch.randint(0, env.action_dim, (env.num_agents,), device=env.device)
        else:
            action = torch.rand((env.num_agents, env.action_dim), device=env.device)
            
        state, reward, done = env.step(action)
        total_reward += reward.sum().item()
        
        plt.clf()
        plt.xlim(0, env.env_size[0].item())
        plt.ylim(0, env.env_size[1].item())
        
        plt.scatter(env.goal_pos[:,0].cpu(), env.goal_pos[:,1].cpu(), c='green', marker='*', s=200, label='Goals')
        plt.scatter(env.obs_pos[:,0].cpu(), env.obs_pos[:,1].cpu(), c='red', marker='X', s=100, label='Obstacles')
        plt.scatter(env.state[:,0].cpu(), env.state[:,1].cpu(), c='blue', marker='o', s=100, label='Agents')
        
        plt.legend()
        print(f"State: {env.state.cpu().numpy()}, Reward: {reward.squeeze(1).cpu().numpy()}, Done: {done.squeeze(1).cpu().numpy()}")
        plt.pause(0.1)