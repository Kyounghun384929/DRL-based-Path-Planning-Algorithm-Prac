import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import os
import time

from src.envs.env_2d import Simple2DGridENV
from src.envs.env_3d import Simple3DGridENV
from src.algorithms import DQNAgent

# Optional: Custom style if available
try:
    from kkh_utils import apply_research_style
    apply_research_style()
except ImportError:
    pass

def get_args():
    parser = argparse.ArgumentParser(description="Test RL Agent and Visualize Path")
    
    # Environment settings
    parser.add_argument("--env_name", type=str, default="2d", choices=["2d", "3d"], help="Environment type")
    parser.add_argument("--max_steps", type=int, default=200, help="Max steps per episode")
    
    # Algorithm settings
    parser.add_argument("--algo", type=str, default="dqn", choices=["dqn", "ppo"], help="Algorithm to use")
    parser.add_argument("--use_noisy", action="store_true", help="Use NoisyNet for DQN (must match training)")
    
    # Model loading
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file (.pth)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    
    return parser.parse_args()

def visualize_path_2d(path, start_pos, goal_pos, env_size, gif_path, png_path):
    path = np.array(path)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, env_size[0])
    ax.set_ylim(0, env_size[1])
    ax.grid(True)
    
    # Plot start and goal
    ax.scatter(start_pos[0], start_pos[1], c='g', s=100, marker='o', label='Start')
    ax.scatter(goal_pos[0], goal_pos[1], c='r', s=100, marker='x', label='Goal')
    
    # Initialize plot elements
    line, = ax.plot([], [], 'b-', alpha=0.6, label='Path')
    point, = ax.plot([], [], 'bo', markersize=5)
    
    ax.set_title("Agent Path in 2D Environment")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    
    def update(frame):
        line.set_data(path[:frame+1, 0], path[:frame+1, 1])
        point.set_data([path[frame, 0]], [path[frame, 1]]) # Pass as sequence
        return line, point
    
    # Save GIF
    print(f"Saving GIF to {gif_path}...")
    # Optimization: Limit max frames to speed up saving
    max_frames = 60
    step = max(1, len(path) // max_frames)
    frames = range(0, len(path), step)
    
    # Ensure last frame is included
    if len(path) - 1 not in frames:
        frames = list(frames)
        frames.append(len(path) - 1)
        
    anim = FuncAnimation(fig, update, frames=frames, interval=100, blit=True)
    anim.save(gif_path, writer='pillow', fps=10, dpi=80)
    print("GIF saved.")

    # Save PNG (High DPI)
    print(f"Saving PNG to {png_path}...")
    # For static display, just plot everything
    ax.plot(path[:, 0], path[:, 1], 'b-', alpha=0.6)
    ax.scatter(path[:, 0], path[:, 1], c='b', s=10, alpha=0.6)
    plt.savefig(png_path, dpi=600)
    print("PNG saved.")
    plt.close(fig)

def visualize_path_3d(path, start_pos, goal_pos, env_min, env_max, gif_path, png_path):
    path = np.array(path)
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set limits
    ax.set_xlim(env_min[0], env_max[0])
    ax.set_ylim(env_min[1], env_max[1])
    ax.set_zlim(env_min[2], env_max[2])
    
    # Plot start and goal
    ax.scatter(start_pos[0], start_pos[1], start_pos[2], c='g', s=100, marker='o', label='Start')
    ax.scatter(goal_pos[0], goal_pos[1], goal_pos[2], c='r', s=100, marker='x', label='Goal')
    
    # Initialize plot elements
    line, = ax.plot([], [], [], 'b-', alpha=0.6, label='Path')
    point, = ax.plot([], [], [], 'bo', markersize=5)
    
    ax.set_title("Agent Path in 3D Environment")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    
    def update(frame):
        line.set_data(path[:frame+1, 0], path[:frame+1, 1])
        line.set_3d_properties(path[:frame+1, 2])
        
        point.set_data([path[frame, 0]], [path[frame, 1]]) # Pass as sequence
        point.set_3d_properties([path[frame, 2]]) # Pass as sequence
        return line, point
    
    # Save GIF
    print(f"Saving GIF to {gif_path}...")
    # Optimization: Limit max frames to speed up saving
    max_frames = 60
    step = max(1, len(path) // max_frames)
    frames = range(0, len(path), step)
    
    # Ensure last frame is included
    if len(path) - 1 not in frames:
        frames = list(frames)
        frames.append(len(path) - 1)

    anim = FuncAnimation(fig, update, frames=frames, interval=100, blit=False) # blit=False for 3D usually safer
    anim.save(gif_path, writer='pillow', fps=10, dpi=80)
    print("GIF saved.")

    # Save PNG (High DPI)
    print(f"Saving PNG to {png_path}...")
    # For static display
    ax.plot(path[:, 0], path[:, 1], path[:, 2], 'b-', alpha=0.6)
    ax.scatter(path[:, 0], path[:, 1], path[:, 2], c='b', s=10, alpha=0.6)
    plt.savefig(png_path, dpi=600)
    print("PNG saved.")
    plt.close(fig)

def main():
    args = get_args()
    device = torch.device(args.device)
    
    print(f"Loading environment: {args.env_name}")
    if args.env_name == "2d":
        env = Simple2DGridENV(state_dim=2, action_dim=4, max_episode_steps=args.max_steps, device=device)
        state_dim = 2
        action_dim = 4
    elif args.env_name == "3d":
        env = Simple3DGridENV(state_dim=3, action_dim=6, max_episode_steps=args.max_steps, device=device)
        state_dim = 3
        action_dim = 6
        
    print(f"Initializing agent: {args.algo}")
    if args.algo == "dqn":
        agent = DQNAgent(
            state_dim=state_dim, 
            action_dim=action_dim, 
            use_noisy=args.use_noisy, 
            device=device,
            epsilon=0.0 # Test mode: greedy action
        )
    elif args.algo == "ppo":
        raise NotImplementedError("PPO not implemented yet")
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    if not torch.cuda.is_available():
        checkpoint = torch.load(args.model_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(args.model_path)
        
    agent.q_net.load_state_dict(checkpoint['model_state_dict'])
    agent.q_net.eval() # Set to evaluation mode
    
    # Run episode
    print("Running episode...")
    state = env.reset()
    done = False
    path = [state.cpu().numpy()]
    total_reward = 0
    steps = 0
    
    while not done:
        # Force greedy action for testing (epsilon=0 is set in init, but just to be sure for noisy net)
        # For NoisyNet, we might want to remove noise, but standard practice is just eval()
        action = agent.get_action(state)
        
        next_state, reward, done = env.step(action)
        
        path.append(next_state.cpu().numpy())
        total_reward += reward
        steps += 1
        state = next_state
        
        print(f"Step: {steps}, Action: {action}, Reward: {reward:.4f}")

    print(f"Episode finished. Total Reward: {total_reward:.4f}, Steps: {steps}")
    
    # Prepare save paths
    save_dir = os.path.join("db", "images", args.env_name, args.algo)
    os.makedirs(save_dir, exist_ok=True)
    
    gif_filename = f"{args.env_name}_path.gif"
    png_filename = f"{args.env_name}_path.png"
    
    gif_path = os.path.join(save_dir, gif_filename)
    png_path = os.path.join(save_dir, png_filename)

    # Visualization
    if args.env_name == "2d":
        visualize_path_2d(
            path, 
            env.init_pos.cpu().numpy(), 
            env.goal_pos.cpu().numpy(), 
            env.env_size.cpu().numpy(),
            gif_path=gif_path,
            png_path=png_path
        )
    elif args.env_name == "3d":
        visualize_path_3d(
            path, 
            env.init_pos.cpu().numpy(), 
            env.goal_pos.cpu().numpy(), 
            env.env_min.cpu().numpy(), 
            env.env_max.cpu().numpy(),
            gif_path=gif_path,
            png_path=png_path
        )

if __name__ == "__main__":
    main()

