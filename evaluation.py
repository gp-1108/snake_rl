"""
Author: gp-1108
Name: evaluation.py
Description: This file contains the evaluation code for the model. This is effectively a copy of the training code from the
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

# Hyperparameters
MOVES_PER_GAME = 1000
NUM_BOARDS = 1000
BOARD_SIZE = 7

# Disable interactive logging
tf.keras.utils.disable_interactive_logging()

# Initialize environments and agents
base_env = OriginalSnakeEnvironment(NUM_BOARDS, BOARD_SIZE)
hybrid_env = OriginalSnakeEnvironment(NUM_BOARDS, BOARD_SIZE)

hybrid_agent_test = HybridDQNAgent(hybrid_env.boards)
hybrid_agent_test.model.load_weights("saved_models/hybrid_model.weights.h5")
baseline_agent_test = BaselineAgent()

# Initialize variables for tracking fruits eaten
hybrid_fruits_list = []
baseline_fruits_list = []

hybrid_wall_hits = []
baseline_wall_hits = []

hybrid_ate_himself_list = []
baseline_ate_himself_list = []

hybrid_fruits = 0
baseline_fruits = 0

hybrid_wall = 0
baseline_wall = 0

hybrid_ate_himself = 0
baseline_ate_himself = 0

for step in trange(MOVES_PER_GAME):
    # Get actions from the agents
    hybrid_actions = hybrid_agent_test.get_actions(hybrid_env.boards)
    baseline_actions = baseline_agent_test.get_actions(base_env.boards)
    
    # Perform the actions and get the rewards from the environment
    hybrid_reward = hybrid_env.move(hybrid_actions)
    baseline_reward = base_env.move(baseline_actions)
    
    # The fruit reward is 0.5, count the number of fruits eaten
    hybrid_fruits += np.sum(hybrid_reward == 0.5)
    baseline_fruits += np.sum(baseline_reward == 0.5)

    # The wall hit reward is -0.1, count the number of wall hits
    hybrid_wall += np.sum(hybrid_reward == -0.1)
    baseline_wall += np.sum(baseline_reward == -0.1)

    # The ate himself reward is -0.2, count the number of times the snake ate himself
    hybrid_ate_himself += np.sum(hybrid_reward == -0.2)
    baseline_ate_himself += np.sum(baseline_reward == -0.2)
    
    # Every 10 steps, save the current number of fruits eaten
    if (step + 1) % 10 == 0:
        hybrid_fruits_list.append(hybrid_fruits)
        baseline_fruits_list.append(baseline_fruits)

        hybrid_wall_hits.append(hybrid_wall)
        baseline_wall_hits.append(baseline_wall)

        hybrid_ate_himself_list.append(hybrid_ate_himself)
        baseline_ate_himself_list.append(baseline_ate_himself)

print(f"Hybrid Agent Overall Fruits: {hybrid_fruits}")
print(f"Baseline Agent Overall Fruits: {baseline_fruits}")
print(f"Hybrid Agent Overall Wall Hits: {hybrid_wall}")
print(f"Baseline Agent Overall Wall Hits: {baseline_wall}")
print(f"Hybrid Agent Overall Ate Himself: {hybrid_ate_himself}")
print(f"Baseline Agent Overall Ate Himself: {baseline_ate_himself}")


# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(range(10, MOVES_PER_GAME+1, 10), hybrid_fruits_list, label='Hybrid Agent Fruits', color='green', alpha=0.7)
plt.plot(range(10, MOVES_PER_GAME+1, 10), baseline_fruits_list, label='Baseline Agent Fruits', color='blue', alpha=0.7)
plt.plot(range(10, MOVES_PER_GAME+1, 10), hybrid_wall_hits, label='Hybrid Agent Wall Hits', color='red', alpha=0.7)
plt.plot(range(10, MOVES_PER_GAME+1, 10), baseline_wall_hits, label='Baseline Agent Wall Hits', color='orange', alpha=0.7)
plt.plot(range(10, MOVES_PER_GAME+1, 10), hybrid_ate_himself_list, label='Hybrid Agent Ate Himself', color='purple', alpha=0.7)
plt.plot(range(10, MOVES_PER_GAME+1, 10), baseline_ate_himself_list, label='Baseline Agent Ate Himself', color='black', alpha=0.7)

# Customize the plot
plt.title('Events Hybrid vs. Baseline', fontsize=16)
plt.xlabel('Iteration', fontsize=12)
plt.legend(fontsize=10, loc="center left")
plt.grid(True, alpha=0.3)

# Add text with overall cumulative fruits eaten
plt.text(0.02, 0.98, f"Hybrid Agent Overall Fruits: {hybrid_fruits}", 
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
plt.text(0.02, 0.94, f"Baseline Agent Overall Fruits: {baseline_fruits}",
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
plt.text(0.02, 0.90, f"Hybrid Agent Overall Wall Hits: {hybrid_wall}",
            transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
plt.text(0.02, 0.86, f"Baseline Agent Overall Wall Hits: {baseline_wall}",
            transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
plt.text(0.02, 0.82, f"Hybrid Agent Overall Ate Himself: {hybrid_ate_himself}",
            transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
plt.text(0.02, 0.78, f"Baseline Agent Overall Ate Himself: {baseline_ate_himself}",
            transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

# Show the plot
plt.tight_layout()
plt.show()