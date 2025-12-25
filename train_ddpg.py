from git import Actor
import torch

from src.algorithms import DDPGAgent
from src.envs import Simple2DContinuousENV
from src.networks import DDPG_Actor, DDPG_Critic
from src.utils import OUNoise

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

env = Simple2DContinuousENV(device=DEVICE)
state_dim = env.state_dim
action_dim = env.action_dim

actor = DDPG_Actor(state_dim, action_dim).to(DEVICE)
critic = DDPG_Critic(state_dim, action_dim).to(DEVICE)
agent = DDPGAgent(actor=actor, critic=critic, device=DEVICE)

noise = OUNoise(action_dim, device=DEVICE)

def train(num_episodes=1000):
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        reach = False
        noise.reset()
        
        while not done:
            action = agent.get_action(state, noise=noise)
            
            next_state, reward, done = env.step(action)
            
            done = done.to(DEVICE, dtype=torch.float32)
            
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward.item()
            
            if env.current_step < env.max_episode_steps and done.item() == 1.0:
                reach = True
            
            if done.item():
                break
            
            agent.train_step()
        
        print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward:.3f}, Steps: {env.current_step}/{env.max_episode_steps}, Reach: {reach}, Last State: {state.cpu().numpy()}")
    
    agent.save("ddpg_agent_1.pth")

if __name__ == "__main__":
    train(num_episodes=1000)