from math import e
import os
import csv
import time
import torch
import argparse

from src.envs import get_env
from src.algorithms import PPOAgent
from src.networks import ActorCritic

def get_args():
    parser = argparse.ArgumentParser(description="Train RL Agent")
    
    # Environment settings
    parser.add_argument("--env_name", type=str, default="2d", choices=["2d", "3d"], help="Environment type")
    parser.add_argument("--max_steps", type=int, default=500, help="Max steps per episode")
    
    # Algorithm settings
    parser.add_argument("--lr_actor", type=float, default=3e-5, help="Learning rate for actor")
    parser.add_argument("--lr_critic", type=float, default=1e-4, help="Learning rate for critic")
    
    # Training settings
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    
    # Logging & Saving
    parser.add_argument("--exp_name", type=str, default="default_exp", help="Experiment name")
    parser.add_argument("--save_dir", type=str, default="./db/saves", help="Directory to save models")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory to save logs")
    parser.add_argument("--save_freq", type=int, default=100, help="Save model every N episodes")
    
    return parser.parse_args()

def main():
    args = get_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Init Env
    device = torch.device(args.device)
    
    env = get_env(args.env_name, max_episode_steps=args.max_steps)
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    network = ActorCritic(state_dim, action_dim).to(device)
    agent = PPOAgent(model=network, device=device, lr_actor=args.lr_actor, lr_critic=args.lr_critic)
    
    # logging setup
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Define directories
    save_dir = os.path.join(args.save_dir, "ppo", args.env_name, timestamp)
    log_dir = os.path.join(args.log_dir, "ppo", args.env_name, timestamp)
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    csv_path = os.path.join(log_dir, f"ppo_{args.env_name}_{args.exp_name}_{timestamp}.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["episode", "reward", "steps", "reach", "loss"])
    
    print(f"Starting training ppo on {args.env_name} environement...")
    
    for episode in range(1, args.episodes + 1):
        state          = env.reset()
        episode_reward = 0
        steps          = 0
        loss           = 0
        reach          = False
        done           = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            if env.current_step < env.max_episode_steps and done is True:
                reach = True
            
            if done:
                break
            
        agent.ppo_update()
        
        # Logging
        print(f"Episode {episode}/{args.episodes}, Reward: {episode_reward:.3f}, Steps: {steps}, Reach: {reach}")
        
        csv_writer.writerow([episode, episode_reward, steps, reach, loss])
        csv_file.flush()
        
        # Save model
        
        if episode % args.save_freq == 0:
            save_path = os.path.join(save_dir, f"ep{episode}.pth")
            
            agent.save(save_path)
            print(f"Saved model at {save_path}")
        
    csv_file.close()
    
    # Final save
    final_save_path = os.path.join(save_dir, "final.pth")
    agent.save(final_save_path)
    print(f"Training completed. Final model saved at {final_save_path}")
    
if __name__ == "__main__":
    main()