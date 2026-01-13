import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from datetime import datetime

from src.envs import Env2DMA
from src.algorithms import MAPPOAgent

# Config for saving
SAVE_OPTION = True
SAVE_DIR = "db/images/2d/mappo"
os.makedirs(SAVE_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Preserving apply_research_style from original file
try:
    from kkh_utils import apply_research_style
    apply_research_style()
except ImportError:
    pass

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Number of agents set to 3
env = Env2DMA(num_agents=3, device=device, max_episode_steps=500)

agent = MAPPOAgent(
    n_agents=env.num_agents, 
    state_dim=env.state_dim,
    action_dim=env.action_dim,
    device=device
)

# Load pretrained models
try:
    actor_model = torch.load("db/saves/mappo/2d/mappo_actor_500steps.pth", map_location=device)
    critic_model = torch.load("db/saves/mappo/2d/mappo_critic_500steps.pth", map_location=device)
    agent.actor.load_state_dict(actor_model)
    agent.critic.load_state_dict(critic_model)
    print("Pretrained models loaded successfully.")
except FileNotFoundError:
    print("Warning: Model files not found. Using randomly initialized weights.")

# 1. Run tests until at least 2 agents reach their goals
print("Searching for an episode where at least 2 agents reach the goal...")
num_reached = 0
episode_count = 0

while num_reached < 3:
    episode_count += 1
    state = env.reset()
    done = False
    
    # logging for trajectory plotting
    current_trajectory = {i: [] for i in range(env.num_agents)}
    # Store initial positions
    for i in range(env.num_agents):
        current_trajectory[i].append(state[i].cpu().numpy())

    while not done:
        action, _, _ = agent.get_action(state)
        next_state, reward, dones = env.step(action)
        state = next_state
        for i in range(env.num_agents):
            current_trajectory[i].append(state[i].cpu().numpy())
        
        if dones.all():
            done = True
    
    # Calculate success count (distance < 3.0 as defined in env)
    final_distances = torch.norm(env.state - env.goal_pos, dim=1)
    num_reached = (final_distances < 3.0).sum().item()
    print(f"Episode {episode_count}: {num_reached} agents reached the goal.")
    
    if num_reached >= 3:
        trajectory = current_trajectory
        goal_pos = env.goal_pos.clone().cpu().numpy()
        init_pos = np.array([trajectory[i][0] for i in range(env.num_agents)])
        break
    
    # Safety break to avoid infinite loop
    if episode_count >= 100:
        print("Reached max search episodes (100). Visualizing the last episode instead.")
        trajectory = current_trajectory
        goal_pos = env.goal_pos.clone().cpu().numpy()
        init_pos = np.array([trajectory[i][0] for i in range(env.num_agents)])
        break

print(f"Goal met! Visualizing trajectory from Episode {episode_count}...")

# 2. Animation and Path Visualization
# Prepare trajectory data for animation
max_len = max(len(trajectory[i]) for i in range(env.num_agents))
traj_np = np.zeros((env.num_agents, max_len, 2))
for i in range(env.num_agents):
    points = np.array(trajectory[i])
    traj_np[i, :len(points), :] = points
    # Keep the last position for agents that finished early
    if len(points) < max_len:
        traj_np[i, len(points):, :] = points[-1]

fig, ax = plt.subplots()
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_title(f'Agent Trajectories (Eps {episode_count}, Success: {num_reached})')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.grid(True)
ax.set_aspect('equal')

# Plot start and goal positions
for i in range(env.num_agents):
    ax.scatter(init_pos[i, 0], init_pos[i, 1], marker='s', color='green', s=20, alpha=0.6, label=f'Start {i}' if i==0 else "")
    ax.scatter(goal_pos[i, 0], goal_pos[i, 1], marker='*', color='red', s=50, alpha=0.6, label=f'Goal {i}' if i==0 else "")

lines = [ax.plot([], [], lw=1.0, marker='o', markersize=1, label=f'Agent {i}')[0] for i in range(env.num_agents)]
scatters = [ax.scatter([], [], s=5) for i in range(env.num_agents)]

def init():
    for line, scatter in zip(lines, scatters):
        line.set_data([], [])
        scatter.set_offsets(np.empty((0, 2)))
    return lines + scatters

def update(frame):
    for i in range(env.num_agents):
        # Update path line up to current frame
        lines[i].set_data(traj_np[i, :frame, 0], traj_np[i, :frame, 1])
        # Update current agent position
        if frame > 0:
            scatters[i].set_offsets(traj_np[i, frame-1:frame, :])
            scatters[i].set_color(lines[i].get_color())
    return lines + scatters

# Adjust frames to keep animation fast (target ~50 frames)
target_frames = 50
skip = max(1, max_len // target_frames)
frames = list(range(1, max_len + 1, skip))
if max_len not in frames:
    frames.append(max_len)

# Create animation
ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=30, repeat=True)

plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()

if SAVE_OPTION:
    # 1. Save PNG (Full Trajectory)
    # To save the full path, we momentarily set data to the final frame
    for i in range(env.num_agents):
        lines[i].set_data(traj_np[i, :, 0], traj_np[i, :, 1])
        scatters[i].set_offsets(traj_np[i, -1:, :])
        scatters[i].set_color(lines[i].get_color())
    
    png_path = os.path.join(SAVE_DIR, f"mappo_trajectory_{timestamp}.png")
    fig.savefig(png_path)
    print(f"Trajectory image saved to: {png_path}")
    
    # 2. Save GIF
    gif_path = os.path.join(SAVE_DIR, f"mappo_trajectory_{timestamp}.gif")
    print(f"Saving GIF ({len(frames)} frames)...")
    # speed up by reducing dpi and adjusting fps
    ani.save(gif_path, writer='pillow', fps=15, dpi=80)
    print(f"Animation saved to: {gif_path}")
    
    # Reset for plt.show()
    init()

plt.show()
