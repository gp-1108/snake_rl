from .BaseAgent import BaseAgent
import numpy as np

class RandomAgent(BaseAgent):
    def __init__(self, boards):
        super().__init__(boards)
    
    def get_actions(self):
        return np.random.choice(range(4), size=(self.boards.shape[0], 1))
    
    def get_action(self, board):
        return np.random.choice(range(4))
    
    def learn(self, actions, rewards):
        pass