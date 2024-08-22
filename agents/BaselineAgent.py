from .BaseAgent import BaseAgent
import numpy as np

class BaselineAgent(BaseAgent):
    def __init__(self, boards):
        super().__init__(boards)
        self.dirs = np.array([(1, 0), (0, 1), (-1, 0), (0, -1)])

    def get_actions(self):
        # Ensure boards is 3D: (num_boards, height, width)
        if self.boards.ndim == 2:
            self.boards = self.boards[np.newaxis, :, :]

        num_boards, height, width = self.boards.shape

        # Get positions of heads and fruits for all boards
        heads = np.array([np.unravel_index(np.argmax(b == self.HEAD), (height, width)) for b in self.boards])
        fruits = np.array([np.unravel_index(np.argmax(b == self.FRUIT), (height, width)) for b in self.boards])

        # Calculate new head positions for all directions and all boards
        new_heads = heads[:, np.newaxis, :] + self.dirs

        # Clip new head positions to be within board boundaries
        new_heads = np.clip(new_heads, 0, np.array([height-1, width-1]))

        # Create masks for illegal moves (walls and body)
        illegal_mask = np.zeros((num_boards, 4), dtype=bool)
        for i in range(num_boards):
            for j, new_head in enumerate(new_heads[i]):
                if (self.boards[i][new_head[0], new_head[1]] == self.WALL or 
                    self.boards[i][new_head[0], new_head[1]] == self.BODY):
                    illegal_mask[i, j] = True

        # Calculate distances for all new head positions to fruits
        distances = np.linalg.norm(new_heads - fruits[:, np.newaxis, :], axis=2)

        # Apply penalty for illegal moves
        distances[illegal_mask] = np.inf

        # Choose the direction with the minimum distance for each board
        actions = np.argmin(distances, axis=1)

        # Check for cases where all moves are illegal
        all_illegal = np.all(illegal_mask, axis=1)
        actions[all_illegal] = self.NONE

        return actions.reshape(-1, 1)
    
    def get_action(self, board):
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

        # If all moves are illegal, return NONE
        if np.all(illegal_mask):
            return self.NONE

        # Choose the direction with the minimum distance
        return np.argmin(distances)