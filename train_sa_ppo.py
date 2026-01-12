import os
import csv
import time
import torch
import argparse
import numpy as np

from src.envs import get_env
from src.algorithms import PPOAgent
from src.networks import ActorCritic

def get_args():
    parser = argparse.ArgumentParser(description="Train RL Agent")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    
    # Environment Settings
    parser.add_argument("--env_name", type=str, default="2d", choices=["2d", "3d"], help="Environment type")
    parser.add_argument("--max_steps", type=int, default=200, help="Max steps per episode")
    
    # Training Settings
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--save_freq", type=int, default=100, help="Save model every N episodes")
    
    # Algorithm Settings
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--eps_clip", type=float, default=0.2, help="Clipping epsilon for PPO")
    parser.add_argument("--K_epochs", type=int, default=20, help="Number of epochs per update")
    
    return parser.parse_args()

def main():
    args = get_args()
    
    # Set seeds
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    
    device = torch.device(args.device)
    
    env = get_env(args.env_name, max_episode_steps=args.max_steps, device=device)
    
    state_dim = 2 if args.env_name == "2d" else 3
    action_dim = 4 if args.env_name == "2d" else 6
    ppo_net = ActorCritic(state_dim, action_dim).to(device)
    agent = PPOAgent(
        model=ppo_net,
        lr=args.lr, 
        gamma=args.gamma, 
        eps_clip=args.eps_clip, 
        K_epochs=args.K_epochs, 
        device=device
    )
    
    # --- Logging and Saving Setup ---
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # 모델 저장 경로: db/saves/ppo/<timestamp>/
    save_dir = os.path.join("./db/saves", "ppo", args.env_name, timestamp)
    log_dir = os.path.join("./logs", "ppo", args.env_name)
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # CSV 파일 열기 (쓰기 모드)
    csv_path = os.path.join(log_dir, f"ppo_{args.env_name}_{timestamp}.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    
    # 헤더 작성
    csv_writer.writerow(["episode", "reward", "steps", "loss"])
    csv_file.flush()
    
    for episode in range(1, args.episodes + 1):
        state = env.reset()
        
        done           = False
        episode_reward = 0
        steps          = 0
        loss           = 0
        
        while not done:
            if isinstance(state, np.ndarray):
                state_input = torch.FloatTensor(state).to(device)
            else:
                state_input = state.to(device)

            action, prob_a, _ = agent.policy.act(state_input)
            
            action = action.item()
            prob_a = prob_a.item()
            
            next_state, reward, done = env.step(action)
            
            if isinstance(state, torch.Tensor):
                state = state.cpu()
            else:
                state = torch.tensor(state)
                
            if isinstance(next_state, torch.Tensor):
                next_state = next_state.cpu()
            else:
                next_state = torch.tensor(next_state)
            
            # Rollout Buffer
            agent.put_data((state, action, reward, next_state, prob_a, done))
            
            state = next_state
            episode_reward += reward
            steps += 1
            
        train_loss = agent.train_step()
        if train_loss is not None:
            loss = train_loss
        
        # Logging
        avg_loss = loss
        
        print(f"Episode {episode} | Reward: {episode_reward:.2f} | Steps: {steps} | Loss: {avg_loss:.4f}")
        
        csv_writer.writerow([episode, episode_reward, steps, avg_loss])
        csv_file.flush() # 강제 종료 대비 즉시 디스크 기록
        
        if episode % args.save_freq == 0: # args.save_freq 사용 권장
            save_path = os.path.join(save_dir, f"ep{episode}.pth")
            torch.save(agent.policy.state_dict(), save_path)
            print(f"Model saved to {save_path}")
        
    
    # End of training
    csv_file.close() # 파일 닫기
    
    # Save final model
    final_save_path = os.path.join(save_dir, f"final.pth")
    torch.save(agent.policy.state_dict(), final_save_path)
    print(f"Final model saved to {final_save_path}")

if __name__ == "__main__":
    main()