from tabnanny import check
import torch
import matplotlib.pyplot as plt

from kkh_utils import apply_research_style

apply_research_style()

def render_2d_path(env, path, save_path=None):
    plt.figure()
    plt.xlim(0, env.env_size[0].item())
    plt.ylim(0, env.env_size[1].item())
    
    # Plot environment boundaries
    plt.gca().add_patch(plt.Rectangle((0, 0), env.env_size[0].item(), env.env_size[1].item(),
                                    fill=None, edgecolor='black', linewidth=2))
    
    # Plot obstacle
    obs = plt.Circle((env.obs_pos[0].item(), env.obs_pos[1].item()), 5.0, color='red', alpha=0.5)
    plt.gca().add_patch(obs)
    
    # Plot start and goal positions
    plt.scatter(env.init_pos[0].item(), env.init_pos[1].item(), color='green', s=100, label='Start')
    plt.scatter(env.goal_pos[0].item(), env.goal_pos[1].item(), color='blue', s=100, label='Goal')
    
    # Plot path
    path = torch.stack(path).cpu().numpy()
    plt.plot(path[:,0], path[:,1], color='orange', linewidth=2, label='Path')
    
    plt.legend()
    plt.title('2D Path Planning')
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()
    

if __name__ == "__main__":
    from src.envs import UAVEnv2D
    # from src.drl.dqn import DQNAgent
    from src.drl.algorithm import PPOAgent
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # env_type = "discrete"
    env_type = "continuous"
    env = UAVEnv2D(env_type=env_type, device=device)
    
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    # Epsilon을 0.0으로 설정하여 무작위 탐험 비활성화 (순수 신경망 추론)
    agent = PPOAgent(device=device, env_type=env_type, state_dim=state_dim, action_dim=action_dim)
    checkpoints_dir = './db/checkpoints/2D/PPO/Continuous/ppo_final.pth'
    agent.policy_old.load_state_dict(torch.load(checkpoints_dir))
    agent.policy.load_state_dict(torch.load(checkpoints_dir))
    
    state = env.reset()
    state /= 100.0
    done = False
    
    path = [env.state[:2].clone()]
    while not done:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        
        state = next_state
        state /= 100.0
        
        path.append(env.state[:2].clone())
    
    render_2d_path(env, path, save_path=None)
    
    