import numpy as np
from .BaselineAgent import BaselineAgent
from .DQNAgent import DQNAgent

class HybridDQNAgent(BaselineAgent, DQNAgent):
    def __init__(self, boards_sample, alpha=0.1, gamma=0.95, epsilon=1.0, decay=0.99, threshold=5):
        BaselineAgent.__init__(self)
        DQNAgent.__init__(self, boards_sample, alpha, gamma, epsilon, decay)
        self.threshold = threshold
        self.update_dirs = np.array([(1, 0), (0, 1), (-1, 0), (0, -1), (0, 0)])

    def get_actions(self, boards, exploration=True):
        if boards.ndim == 2:
            boards = boards[np.newaxis, :, :]
        
        num_boards, height, width = boards.shape

        # Calculate body lengths for all boards
        body_lengths = np.sum(boards == self.BODY, axis=(1, 2)) + 1  # +1 for the head

        # Create masks for baseline and DQN strategies
        baseline_mask = body_lengths < self.threshold
        dqn_mask = ~baseline_mask

        # Initialize actions array
        actions = np.zeros((num_boards, 1), dtype=int)

        # Get actions for boards using baseline strategy
        if np.any(baseline_mask):
            actions[baseline_mask] = BaselineAgent.get_actions(self, boards[baseline_mask])

        # Get actions for boards using DQN strategy
        if np.any(dqn_mask):
            dqn_actions = DQNAgent.get_actions(self, boards[dqn_mask], exploration)
            
            # Avoid wall collisions for DQN actions
            safe_actions = self.avoid_wall_collisions(boards[dqn_mask], dqn_actions)
            actions[dqn_mask] = safe_actions

        return actions

    def avoid_wall_collisions(self, boards, actions):
        actions = np.array(actions)
        num_boards, height, width = boards.shape
        
        # Find head positions
        heads = np.array([np.unravel_index(np.argmax(b == self.HEAD), (height, width)) for b in boards])
        
        # Calculate new head positions based on actions
        new_heads = heads + self.update_dirs[actions.flatten()]
        
        # Check if new head positions are within bounds and not walls
        valid_moves = (
            (0 <= new_heads[:, 0]) & (new_heads[:, 0] < height) &
            (0 <= new_heads[:, 1]) & (new_heads[:, 1] < width) &
            (boards[np.arange(num_boards), new_heads[:, 0], new_heads[:, 1]] != self.WALL)
        )

        # Where moves are invalid, use BaselineAgent
        invalid_mask = ~valid_moves
        if np.any(invalid_mask):
            actions[invalid_mask] = BaselineAgent.get_actions(self, boards[invalid_mask]).reshape(-1, 1)
        
        return actions

    def get_action(self, board):
        return self.get_actions(board[np.newaxis, :, :])[0]

    def learn(self, prev_boards, actions, rewards, next_boards):
        # Only learn for states where DQN was used
        dqn_mask = np.sum(prev_boards == self.BODY, axis=(1, 2)) + 1 >= self.threshold
        
        if np.any(dqn_mask):
            DQNAgent.learn(self, 
                           prev_boards[dqn_mask], 
                           actions[dqn_mask], 
                           rewards[dqn_mask], 
                           next_boards[dqn_mask])