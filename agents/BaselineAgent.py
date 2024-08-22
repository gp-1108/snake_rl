from .BaseAgent import BaseAgent
import numpy as np

class BaselineAgent(BaseAgent):
    HEAD = 4
    BODY = 3
    FRUIT = 2
    EMPTY = 1
    WALL = 0
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    NONE = 4
    def __init__(self, boards):
        super().__init__(boards)
    
    def get_actions(self):
        actions = []
        for i in range(self.boards.shape[0]):
            actions.append(self.get_action(self.boards[i]))
        return np.array(actions).reshape(-1, 1)
    
    def get_action(self, board):
        # Getting the infos from the board
        head = np.argwhere(board == self.HEAD)[0]
        fruit = np.argwhere(board == self.FRUIT)[0]
        bodies = np.argwhere(board == self.BODY)
        walls = np.argwhere(board == self.WALL)

        # Getting the new possible heads positions UP, RIGHT, DOWN, LEFT
        dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        new_heads = [head + np.array(dir) for dir in dirs]


        # Calculate distances while considering illegal moves
        distances = []
        for new_head in new_heads:
            if any(np.array_equal(new_head, w) for w in walls) or any(np.array_equal(new_head, b) for b in bodies):
                distances.append(float('inf'))  # Assign a high cost for illegal moves
            else:
                distances.append(np.linalg.norm(new_head - fruit))

        # Choosing the direction that gets closer to the fruit
        directions = [self.UP, self.RIGHT, self.DOWN, self.LEFT]

        return directions[np.argmin(distances)]
    
    def learn(self, actions, rewards):
        pass