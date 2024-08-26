from .BaseAgent import BaseAgent
import numpy as np

class BaselineAgent(BaseAgent):
    """
    BaselineAgent class represents a simple agent that makes decisions based on the current game board.
    Attributes:
        dirs (numpy.ndarray): Array of directions representing possible moves.
    Methods:
        get_actions(boards):
            Takes a 3D array of game boards and returns the optimal action for each board.
        get_action(board):
            Takes a 2D array representing a single game board and returns the optimal action.
    """
    def __init__(self):
        self.dirs = np.array([(1, 0), (0, 1), (-1, 0), (0, -1)])

    def get_actions(self, boards):
        """
        Calculates the actions to be taken by the agent based on the given game boards.
        Args:
            boards (ndarray): The game boards. Should be 3D with shape (num_boards, height, width).
        Returns:
            ndarray: The actions to be taken by the agent. Each action corresponds to a board and is represented as an integer.
        Raises:
            None
        """
        # Ensure boards is 3D: (num_boards, height, width)
        if boards.ndim == 2:
            boards = boards[np.newaxis, :, :]

        num_boards, height, width = boards.shape

        # Get positions of heads and fruits for all boards
        heads = np.array([np.unravel_index(np.argmax(b == self.HEAD), (height, width)) for b in boards])
        fruits = np.array([np.unravel_index(np.argmax(b == self.FRUIT), (height, width)) for b in boards])

        # Calculate new head positions for all directions and all boards
        new_heads = heads[:, np.newaxis, :] + self.dirs

        # Clip new head positions to be within board boundaries
        new_heads = np.clip(new_heads, 0, np.array([height-1, width-1]))

        # Create masks for illegal moves (walls and body)
        illegal_mask = np.zeros((num_boards, 4), dtype=bool)
        wall_mask = np.zeros((num_boards, 4), dtype=bool)
        for i in range(num_boards):
            for j, new_head in enumerate(new_heads[i]):
                if boards[i][new_head[0], new_head[1]] == self.WALL:
                    illegal_mask[i, j] = True
                    wall_mask[i, j] = True
                elif boards[i][new_head[0], new_head[1]] == self.BODY:
                    illegal_mask[i, j] = True

        # Calculate distances for all new head positions to fruits
        distances = np.linalg.norm(new_heads - fruits[:, np.newaxis, :], axis=2)

        # Apply penalty for illegal moves
        distances[illegal_mask] = np.inf

        # Choose the direction with the minimum distance for each board
        actions = np.argmin(distances, axis=1)

        # Check for cases where all moves are illegal
        all_illegal = np.all(illegal_mask, axis=1)
        
        # For boards where all moves are illegal, choose a random non-wall move
        for i in np.where(all_illegal)[0]:
            non_wall_moves = np.where(~wall_mask[i])[0]
            if len(non_wall_moves) > 0:
                actions[i] = np.random.choice(non_wall_moves)
            else:
                actions[i] = self.NONE  # If all moves lead to walls, choose NONE as a last resort

        return actions.reshape(-1, 1)
    
    def get_action(self, board):
        """
        Calculates the action to take based on the current state of the board.

        Parameters:
        - board (numpy.ndarray): The game board represented as a 2D numpy array.

        Returns:
        - int: The index of the action to take.

        Raises:
        - None

        """
        height, width = board.shape

        # Get positions of head and fruit
        head = np.unravel_index(np.argmax(board == self.HEAD), (height, width))
        fruit = np.unravel_index(np.argmax(board == self.FRUIT), (height, width))

        # Calculate new head positions for all directions
        new_heads = np.array(head) + self.dirs

        # Clip new head positions to be within board boundaries
        new_heads = np.clip(new_heads, 0, np.array([height-1, width-1]))

        # Create mask for illegal moves (walls and body)
        illegal_mask = np.array([
            board[new_head[0], new_head[1]] == self.WALL or 
            board[new_head[0], new_head[1]] == self.BODY
            for new_head in new_heads
        ])

        # Calculate distances for all new head positions to fruit
        distances = np.linalg.norm(new_heads - fruit, axis=1)

        # Apply penalty for illegal moves
        distances[illegal_mask] = np.inf

        # If all moves are illegal, return a random move that is not a wall
        if np.all(illegal_mask):
            non_wall_moves = np.where(board != self.WALL)[0]
            return np.random.choice(non_wall_moves)

        # Choose the direction with the minimum distance
        return np.argmin(distances)