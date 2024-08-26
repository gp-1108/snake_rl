from .BaseAgent import BaseAgent
import numpy as np

class RandomAgent(BaseAgent):
    """
    RandomAgent is an agent that selects actions randomly.
    Attributes:
        output_size (int): The number of possible actions.
    Methods:
        get_actions(boards): Returns a random action for each board in the input.
        get_action(): Returns a random action.
    """
    def __init__(self):
        """
        Initializes a RandomAgent object.

        Parameters:
        None

        Returns:
        None
        """
        self.output_size = 5 # UP, DOWN, LEFT, RIGHT, NONE
    
    def get_actions(self, boards):
        """
        Generates random actions for the given boards.

        Parameters:
        - boards: numpy.ndarray
            The input boards for which actions need to be generated.

        Returns:
        - numpy.ndarray
            Randomly generated actions for the given boards.

        """
        if boards.ndim == 2:
            boards = boards[np.newaxis, :, :]
        return np.random.choice(range(self.output_size), size=(boards.shape[0], 1))
    
    def get_action(self):
        """
        Generates a random action.

        Returns:
            int: The randomly chosen action.
        """
        return np.random.choice(range(self.output_size))