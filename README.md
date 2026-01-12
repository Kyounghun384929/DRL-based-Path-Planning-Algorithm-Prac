# Reinforcement Learning-based Path Planning Algorithms Repository

This repository aims to implement path planning algorithms based on deep reinforcement learning (DRL).

It addresses the problem of an agent navigating to a goal in simple 2D and 3D grid environments.  
Currently, agents are trained using the DQN (Deep Q-Network) and PPO (Proximal Policy Optimization) algorithms.

# Planned Algorithm Additions

- Single-Agent
    - [x] DQN (Deep Q-Network)   (Done 2025-12-23)
    - [x] PPO (Proximal Policy Optimization) (Done 2025-12-24)
    - [ ] A3C (Asynchronous Advantage Actor-Critic)
    - [x] DDPG (Deep Deterministic Policy Gradient) (Almost done 2025-12-25, but there need some fixes)
    - [ ] SAC (Soft Actor-Critic)
    
- Multi-Agent
    - [x] IQL (Independent Q-Learning) (Done 2026-01-10)
    - [x] VDN (Value Decomposition Networks) (Done 2026-01-12)
    - [ ] MADDPG (Multi-Agent Deep Deterministic Policy Gradient)
    - [x] QMIX (Done 2026-01-12)
    - [ ] COMA (Counterfactual Multi-Agent Policy Gradients)


# Key Files
- `src/envs/`: Implementations of 2D and 3D grid environments
- `src/algorithms/`: Implementations of reinforcement learning algorithms
- `train.py`: Train agents
- `test.py`: Evaluate and visualize trained agents

# Usage
1. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
2. Train an agent
   ```bash
   python train_*.py
   ```
   Replace `*` with the desired algorithm (e.g., `dqn`, `ppo`).
   
3. Evaluate an agent
   ```bash
   python test.py --model_path path_to_trained_model.pth
   ```

# Simulation Results

## 2D Environment

### DQN

Model Path: db\saves\dqn\2d\20251223_141101\final.pth  
Settings:
- max_steps=200
- max_episodes=1000

![2D_DQN](docs/2D-dqn.gif)

### PPO
Model Path: db\saves\ppo\2d\ppo_agent_best.pth  
Settings:
- max_steps=500
- max_episodes=1000

![2D_PPO](docs/2D-ppo.gif)

### DDPG
Model Path: db\saves\ddpg\2d\ddpg_final.pth  
Settings:
- max_steps=200
- max_episodes=1000

![2D_DDPG](docs/2D-ddpg.gif)

# Thanks to  
...

Kyounghun Kim  
sdrudgnsdl@kw.ac.kr  
Seoul, South Korea  
Kwangwoon Univ.