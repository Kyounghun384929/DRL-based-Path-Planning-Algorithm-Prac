import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import time

from src.envs import Simple2DContinuousENV
from src.algorithms import DDPGAgent
from src.networks import DDPG_Actor, DDPG_Critic
from src.utils import OUNoise

# Configuration
MODEL_PATH = "db/saves/ddpg/2d/ddpg_actor_critic_1.pth"  # Path to the trained model
ENV_NAME = "2d"
MAX_STEPS = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GIF_PATH = "ddpg_test_result.gif"
PNG_PATH = "ddpg_test_result.png"

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
    
    ax.set_title("Agent Path in 2D Continuous Environment (DDPG)")
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

def main():
    # 1. Initialize Environment
    env = Simple2DContinuousENV(device=DEVICE)
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    print(f"Environment: {ENV_NAME}, State Dim: {state_dim}, Action Dim: {action_dim}")

    # 2. Initialize Agent & Load Model
    actor = DDPG_Actor(state_dim, action_dim).to(DEVICE)
    critic = DDPG_Critic(state_dim, action_dim).to(DEVICE)
    agent = DDPGAgent(actor=actor, critic=critic, device=DEVICE)
    
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        # Check if checkpoint is a full agent state or just actor state
        if isinstance(checkpoint, dict) and 'actor_state_dict' in checkpoint:
             agent.actor.load_state_dict(checkpoint['actor_state_dict'])
             agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        elif isinstance(checkpoint, dict):
             # Assuming it might be just the actor state dict directly or some other format
             # Try loading into actor if keys match, otherwise warn
             try:
                 agent.actor.load_state_dict(checkpoint)
             except:
                 print("Warning: Could not load state dict directly. Check checkpoint format.")
        else:
             # If it's the entire object saved (not recommended but possible)
             pass
             
        print("Model loaded.")
    else:
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    # 3. Run Test Episode
    state = env.reset()
    done = False
    path = [state.cpu().numpy()]
    total_reward = 0
    step_count = 0
    noise = OUNoise(action_dim, device=DEVICE)
    
    print("Starting test episode...")
    start_time = time.time()
    noise.reset()
    while not done and step_count < MAX_STEPS:
        # Get action (With noise for testing)
        action = agent.get_action(state, noise=noise)
        
        next_state, reward, done = env.step(action)
        
        state = next_state
        path.append(state.cpu().numpy())
        total_reward += reward.item()
        step_count += 1
        
    end_time = time.time()
    print(f"Episode finished. Steps: {step_count}, Total Reward: {total_reward:.3f}, Time: {end_time - start_time:.2f}s")
    
    # 4. Visualize
    start_pos = env.init_pos.cpu().numpy()
    goal_pos = env.goal_pos.cpu().numpy()
    env_size = env.env_size.cpu().numpy()
    
    visualize_path_2d(path, start_pos, goal_pos, env_size, GIF_PATH, PNG_PATH)

if __name__ == "__main__":
    main()
