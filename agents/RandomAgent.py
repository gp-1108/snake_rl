from .BaseAgent import BaseAgent
import numpy as np

class RandomAgent(BaseAgent):
    def __init__(self):
        self.output_size = 5 # UP, DOWN, LEFT, RIGHT, NONE
    
    def get_actions(self, boards):
        if boards.ndim == 2:
            boards = boards[np.newaxis, :, :]
        return np.random.choice(range(self.output_size), size=(boards.shape[0], 1))
    
    def get_action(self):
        return np.random.choice(range(self.output_size))