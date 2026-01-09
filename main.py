import torch
import torch.nn as nn
from environment import EVChargingEnv  
from agent import DQNAgent
from visualizer import EVVisualizer
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import csv
import os

# --- 1. Configuration ---
NUM_EVS = 2
NUM_STATIONS = 2
STATE_DIM = 5  # [x, y, soc, dist_to_s1, dist_to_s2]
ACTION_DIM = NUM_STATIONS
EPISODES = 500
STEPS_PER_EPISODE = 20  # Increased to allow for charging time
BATCH_SIZE = 32  

# --- 2. Data Logger Setup ---
log_file = "ev_marl_results.csv"
if not os.path.exists(log_file):
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Total_Reward", "Avg_SoC", "Epsilon"])

# --- 3. Initialization ---
env = EVChargingEnv(num_evs=NUM_EVS)
agents = [DQNAgent(STATE_DIM, ACTION_DIM) for _ in range(NUM_EVS)]
viz = EVVisualizer(env)

# Initial environment reset
obs = env.reset()
episode_count = 0
current_episode_rewards = []

def update(frame):
    global obs, episode_count, current_episode_rewards
    
    # Check if this is the last step of the episode
    is_done = (frame + 1) % STEPS_PER_EPISODE == 0
    
    # a. Agents decide on actions
    actions = [agents[i].act(obs[i]) for i in range(NUM_EVS)]
    
    # b. Environment takes a step
    next_obs, rewards = env.step(actions)
    
    # c. Agents "Remember" and "Learn"
    for i in range(NUM_EVS):
        # Pass the is_done flag so the agent knows the episode is ending
        agents[i].remember(obs[i], actions[i], rewards[i], next_obs[i], is_done)
        
        # Memory Guard: Only train if we have enough samples to avoid crash
        if len(agents[i].memory) >= BATCH_SIZE:
            agents[i].replay() 
        
    obs = next_obs
    total_step_reward = sum(rewards)
    viz.reward_history.append(total_step_reward)
    current_episode_rewards.append(total_step_reward)
    
    # d. Handle Episode resets and Data Logging
    if is_done:
        ep_reward = sum(current_episode_rewards)
        avg_soc = sum(env.ev_soc) / NUM_EVS
        
        # Log to CSV
        with open(log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode_count, round(ep_reward, 2), round(avg_soc, 3), round(agents[0].epsilon, 3)])
        
        # Decay Epsilon once per episode for stability
        for a in agents:
            if a.epsilon > a.epsilon_min:
                a.epsilon *= a.epsilon_decay

        # Reset for next episode
        current_episode_rewards = []
        obs = env.reset()
        episode_count += 1
        
        # Progress update every 10 episodes
        if episode_count % 10 == 0:
            for a in agents: a.update_target()
            print(f"Episode {episode_count} | Ep Reward: {ep_reward:.2f} | Epsilon: {agents[0].epsilon:.3f}")

    # e. Visual Update
    viz.update(frame, None)

# --- 4. Run Training with Visualization ---
print(f"Starting MARL Training. Results will be saved to {log_file}")

ani = FuncAnimation(viz.fig, update, frames=EPISODES * STEPS_PER_EPISODE, repeat=False, interval=50)

# To save a GIF of the learning process
ani.save("marl_ev_routing.gif", writer=PillowWriter(fps=10))

plt.show()

# Save the trained model
torch.save(agents[0].policy_net.state_dict(), "ev_model_weights.pth")
print("Model saved successfully!")

