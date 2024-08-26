"""
Author: gp-1108
Name: training.py
Description: This file contains the training code for the model. This is effectively a copy of the training code from the
main.ipynb notebook, but in a more modular format.
"""
from environments_fully_observable import OriginalSnakeEnvironment
import numpy as np
from  tqdm import trange
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from agents import BaselineAgent, RandomAgent, DQNAgent, HybridDQNAgent

# Fix random seeds for reproducibility
tf.random.set_seed(0) 
random.seed(0)
np.random.seed(0)

# Training hyperparameters
MOVES_PER_GAME = 1000
NUM_BOARDS = 1000
BOARD_SIZE = 7

tf.keras.utils.disable_interactive_logging() # Removing the annoying tf logging

# Set up the environment and agents
rnd_env = OriginalSnakeEnvironment(NUM_BOARDS, BOARD_SIZE)
base_env = OriginalSnakeEnvironment(NUM_BOARDS, BOARD_SIZE)
dqn_env = OriginalSnakeEnvironment(NUM_BOARDS, BOARD_SIZE)
hybrid_env = OriginalSnakeEnvironment(NUM_BOARDS, BOARD_SIZE)

# Set up the agents
rnd_agent = RandomAgent()
baseline_agent = BaselineAgent()
dqn_agent = DQNAgent(dqn_env.boards)
hybrid_agent = HybridDQNAgent(hybrid_env.boards)

# Store the rewards for each move
rnd_rewards = []
baseline_rewards = []
dqn_rewards = []
hybrid_rewards = []

for _ in trange(MOVES_PER_GAME):
    # Get actions from the agents
    rnd_actions = rnd_agent.get_actions(rnd_env.boards)
    base_actions = baseline_agent.get_actions(base_env.boards)
    dqn_actions = dqn_agent.get_actions(dqn_env.boards)
    hybrid_actions = hybrid_agent.get_actions(hybrid_env.boards)

    dqn_prev_boards = dqn_env.boards.copy()
    hybrid_prev_boards = hybrid_env.boards.copy()
    
    # Perform the actions and get the rewards from the environment
    rnd_reward = rnd_env.move(rnd_actions)
    baseline_reward = base_env.move(base_actions)
    dqn_reward = dqn_env.move(dqn_actions)
    hybrid_reward = hybrid_env.move(hybrid_actions)

    # Learn from it
    dqn_agent.learn(dqn_prev_boards, dqn_actions, dqn_reward, dqn_env.boards)
    hybrid_agent.learn(hybrid_prev_boards, hybrid_actions, hybrid_reward, hybrid_env.boards)

    
    # Store the rewards for each move
    rnd_rewards.append(np.mean(rnd_reward))
    baseline_rewards.append(np.mean(baseline_reward))
    dqn_rewards.append(np.mean(dqn_reward))
    hybrid_rewards.append(np.mean(hybrid_reward))

# Plot the rewards
plt.figure(figsize=(12, 6))
plt.plot(range(1, MOVES_PER_GAME + 1), rnd_rewards, label='Random Agent', color='red', alpha=0.7)
plt.plot(range(1, MOVES_PER_GAME + 1), baseline_rewards, label='Baseline Agent', color='blue', alpha=0.7)
plt.plot(range(1, MOVES_PER_GAME + 1), dqn_rewards, label='DQN Agent', color='green', alpha=0.7)
plt.plot(range(1, MOVES_PER_GAME + 1), hybrid_rewards, label='Hybrid DQN Agent', color='purple', alpha=0.7)

# Add a trend line for each agent
z_rnd = np.polyfit(range(1, MOVES_PER_GAME + 1), rnd_rewards, 1)
p_rnd = np.poly1d(z_rnd)
plt.plot(range(1, MOVES_PER_GAME + 1), p_rnd(range(1, MOVES_PER_GAME + 1)), "r--", alpha=0.5)

z_base = np.polyfit(range(1, MOVES_PER_GAME + 1), baseline_rewards, 1)
p_base = np.poly1d(z_base)
plt.plot(range(1, MOVES_PER_GAME + 1), p_base(range(1, MOVES_PER_GAME + 1)), "b--", alpha=0.5)

z_dqn = np.polyfit(range(1, MOVES_PER_GAME + 1), dqn_rewards, 1)
p_dqn = np.poly1d(z_dqn)
plt.plot(range(1, MOVES_PER_GAME + 1), p_dqn(range(1, MOVES_PER_GAME + 1)), "g--", alpha=0.5)

z_hybrid = np.polyfit(range(1, MOVES_PER_GAME + 1), hybrid_rewards, 1)
p_hybrid = np.poly1d(z_hybrid)
plt.plot(range(1, MOVES_PER_GAME + 1), p_hybrid(range(1, MOVES_PER_GAME + 1)), "purple", alpha=0.5)

# Customize the plot
plt.title('Average Rewards per Iteration: all agents', fontsize=16)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Average Reward', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Add text with overall average rewards
plt.text(0.02, 0.98, f"Random Agent Overall Avg: {np.mean(rnd_rewards):.4f}", 
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
plt.text(0.02, 0.94, f"Baseline Agent Overall Avg: {np.mean(baseline_rewards):.4f}", 
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
plt.text(0.02, 0.90, f"DQN Agent Overall Avg: {np.mean(dqn_rewards):.4f}", 
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
plt.text(0.02, 0.86, f"Hybrid DQN Agent Overall Avg: {np.mean(hybrid_rewards):.4f}",
            transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

# Show the plot
plt.tight_layout()
plt.show()

# Now plot the cumulative rewards
plt.figure(figsize=(12, 6))
plt.plot(range(1, MOVES_PER_GAME + 1), np.cumsum(rnd_rewards), label='Random Agent', color='red', alpha=0.7)
plt.plot(range(1, MOVES_PER_GAME + 1), np.cumsum(baseline_rewards), label='Baseline Agent', color='blue', alpha=0.7)
plt.plot(range(1, MOVES_PER_GAME + 1), np.cumsum(dqn_rewards), label='DQN Agent', color='green', alpha=0.7)
plt.plot(range(1, MOVES_PER_GAME + 1), np.cumsum(hybrid_rewards), label='Hybrid DQN Agent', color='purple', alpha=0.7)

# Customize the plot
plt.title('Cumulative Rewards per Iteration: all agents', fontsize=16)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Cumulative Reward', fontsize=12)
plt.legend(fontsize=10, loc="center left")
plt.grid(True, alpha=0.3)

# Add text with overall cumulative rewards
plt.text(0.02, 0.98, f"Random Agent Overall Cumulative: {np.sum(rnd_rewards):.4f}", 
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
plt.text(0.02, 0.94, f"Baseline Agent Overall Cumulative: {np.sum(baseline_rewards):.4f}",
            transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
plt.text(0.02, 0.90, f"DQN Agent Overall Cumulative: {np.sum(dqn_rewards):.4f}",
            transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
plt.text(0.02, 0.86, f"Hybrid DQN Agent Overall Cumulative: {np.sum(hybrid_rewards):.4f}",
            transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

# Show the plot
plt.tight_layout()
plt.show()

# Save the model
hybrid_model = hybrid_agent.model
hybrid_model.save_weights("hybrid_model.weights.h5")