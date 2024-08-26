import numpy as np
from .BaselineAgent import BaselineAgent
from .DQNAgent import DQNAgent

class HybridDQNAgent(BaselineAgent, DQNAgent):
    """
    A hybrid agent that uses a combination of baseline and DQN strategies to play the game.

    Attributes:
    - threshold (int): The threshold body length for using the DQN strategy.
    - update_dirs (np.array): An array of update directions for each action (UP, RIGHT, DOWN, LEFT, NONE).

    Methods:
    - get_actions(boards, exploration=True): Get the actions for the given boards using a hybrid strategy.
    - avoid_wall_collisions(boards, actions): Avoid wall collisions by checking if the new head positions after taking certain actions are within bounds and not walls.
    - get_action(board): Returns the action to be taken by the agent based on the given board state.
    - learn(prev_boards, actions, rewards, next_boards): Learn from the given experiences.
    - load_model_weights(path): Load the model weights from the specified path.
    """
    def __init__(self, boards_sample, alpha=0.1, gamma=0.95, epsilon=1.0, decay=0.99, threshold=5):
        """
        Initializes a HybridDQNAgent object.

        Parameters:
        - boards_sample (list): A list of game boards used for training.
        - alpha (float): The learning rate for the Q-learning algorithm. Default is 0.1.
        - gamma (float): The discount factor for future rewards in the Q-learning algorithm. Default is 0.95.
        - epsilon (float): The exploration rate for the Q-learning algorithm. Default is 1.0.
        - decay (float): The decay rate for the exploration rate in the Q-learning algorithm. Default is 0.99.
        - threshold (int): The threshold value for updating the agent's direction. Default is 5.
        """
        BaselineAgent.__init__(self)
        DQNAgent.__init__(self, boards_sample, alpha, gamma, epsilon, decay)
        self.threshold = threshold
        self.update_dirs = np.array([(1, 0), (0, 1), (-1, 0), (0, -1), (0, 0)])

    def get_actions(self, boards, exploration=True):
        """
        Get the actions for the given boards using a hybrid strategy.
        It does so by using the baseline strategy for boards with body length less than the threshold,
        and the DQN strategy for boards with body length greater than or equal to the threshold.
        If DQN actions result in wall collisions, the agent uses the baseline strategy instead.

        Parameters:
        - boards (ndarray): The input boards for which to calculate actions of shape (num_boards, height, width).
        - exploration (bool): Flag indicating whether to use exploration during action selection. Default is True.
        Returns:
        - actions (ndarray): The selected actions for the input boards.
        """
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
        """
        Avoids wall collisions by checking if the new head positions after taking certain actions are within bounds and not walls.
        This method is used to correct the actions taken by the DQN agent to avoid wall collisions.

        Parameters:
        - boards (ndarray): Array of shape (num_boards, height, width) representing the game boards.
        - actions (list): List of actions to be taken.
        Returns:
        - actions (ndarray): Array of actions after avoiding wall collisions.
        """
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
        """
        Returns the action to be taken by the agent based on the given board state.

        Parameters:
        - board: (ndarray) The board state.
        Returns: (int) The action to take.
        """
        return self.get_actions(board[np.newaxis, :, :])[0]

    def learn(self, prev_boards, actions, rewards, next_boards):
        """
        Learn from the given experiences.
        Args:
            prev_boards (numpy.ndarray): Array of previous game boards.
            actions (numpy.ndarray): Array of actions taken.
            rewards (numpy.ndarray): Array of rewards received.
            next_boards (numpy.ndarray): Array of next game boards.
        Returns:
            None
        """
        # Only learn for states where DQN was used
        dqn_mask = np.sum(prev_boards == self.BODY, axis=(1, 2)) + 1 >= self.threshold
        
        if np.any(dqn_mask):
            DQNAgent.learn(self, 
                           prev_boards[dqn_mask], 
                           actions[dqn_mask], 
                           rewards[dqn_mask], 
                           next_boards[dqn_mask])
    
    def load_model_weights(self, path):
        """
        Load the model weights from the specified path.

        Parameters:
        - path (str): The path to the model weights file.

        Returns:
        - None
        """
        return DQNAgent.load_model_weights(self, path)