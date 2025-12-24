import os
import csv
import time
import torch
import argparse

# from src.envs.env_2d import Simple2DGridENV
# from src.envs.env_3d import Simple3DGridENV
from src.envs import get_env
# from src.algorithms.dqn import DQNAgent
from src.algorithms import DQNAgent

def get_args():
    parser = argparse.ArgumentParser(description="Train RL Agent")
    
    # Environment settings
    parser.add_argument("--env_name", type=str, default="2d", choices=["2d", "3d"], help="Environment type")
    parser.add_argument("--max_steps", type=int, default=200, help="Max steps per episode")
    
    # Algorithm settings
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--use_noisy", action="store_true", help="Use NoisyNet for DQN")
    
    # Training settings
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Logging & Saving
    parser.add_argument("--exp_name", type=str, default="default_exp", help="Experiment name")
    parser.add_argument("--save_dir", type=str, default="./db/saves", help="Directory to save models")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory to save logs")
    parser.add_argument("--save_freq", type=int, default=100, help="Save model every N episodes")
    
    return parser.parse_args()

def main():
    args = get_args()
    
    # Set seeds
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Initialize Environment
    device = torch.device(args.device)
    env = get_env(args.env_name, state_dim=2 if args.env_name == "2d" else 3, action_dim=4 if args.env_name == "2d" else 6, max_episode_steps=args.max_steps, device=device)
    state_dim = 2 if args.env_name == "2d" else 3
    action_dim = 4 if args.env_name == "2d" else 6
    
    # Initialize Agent
    agent = DQNAgent(
        state_dim=state_dim, 
        action_dim=action_dim, 
        lr=args.lr, 
        use_noisy=args.use_noisy, 
        device=device,
    )

    # Logging setup
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Define directories
    save_dir = os.path.join(args.save_dir, "dqn", args.env_name, timestamp)
    log_dir = os.path.join(args.log_dir, "dqn", args.env_name, timestamp)
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    csv_path = os.path.join(log_dir, f"dqn_{args.env_name}_{args.exp_name}_{timestamp}.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["episode", "reward", "steps", "epsilon", "loss"])
    
    print(f"Start training {args.algo} on {args.env_name} environment...")
    
    for episode in range(1, args.episodes + 1):
        state          = env.reset()
        episode_reward = 0
        steps          = 0
        loss           = 0
        done           = False
        
        while not done:
            action = agent.get_action(state)
            
            next_state, reward, done = env.step(action)
            
            # Store in buffer
            # DQNAgent.replay_buffer.push expects CPU tensors usually to save GPU memory
            if isinstance(state, torch.Tensor):
                state_cpu = state.cpu()
            else:
                state_cpu = torch.tensor(state)
                
            if isinstance(next_state, torch.Tensor):
                next_state_cpu = next_state.cpu()
            else:
                next_state_cpu = torch.tensor(next_state)
                
            agent.replay_buffer.push(state_cpu, action, reward, next_state_cpu, done)
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            # Train
            train_loss = agent.train_step()
            if train_loss is not None:
                loss += train_loss
                
        # End of episode
        agent.update_epsilon()
        
        # Target update (Hardcoded freq for now or add arg)
        if episode % 10 == 0:
            agent.update_target_network()
            
        # Logging
        avg_loss = loss / steps if steps > 0 else 0
        epsilon = agent.epsilon if hasattr(agent, "epsilon") else 0.0
        
        print(f"Ep: {episode}, Reward: {episode_reward:.2f}, Steps: {steps}, Eps: {epsilon:.4f}, Loss: {avg_loss:.4f}")
        
        csv_writer.writerow([episode, episode_reward, steps, epsilon, avg_loss])
        csv_file.flush()
        
        # Save model
        if episode % args.save_freq == 0:
            save_path = os.path.join(save_dir, f"ep{episode}.pth")
            # Save agent state
            torch.save({
                'episode': episode,
                'model_state_dict': agent.q_net.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': epsilon
            }, save_path)
            print(f"Model saved to {save_path}")
    
    
    csv_file.close()
        
    # 학습 종료 시 최종 모델 저장
    final_save_path = os.path.join(save_dir, f"final.pth")
    torch.save({
        'episode': args.episodes,
        'model_state_dict': agent.q_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': epsilon
    }, final_save_path)
    print(f"Final model saved to {final_save_path}")
    
    print("Training finished.")

if __name__ == "__main__":
    main()
