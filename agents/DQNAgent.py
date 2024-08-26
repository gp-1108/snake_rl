from .BaseAgent import BaseAgent
import tensorflow.keras as K
import tensorflow as tf
import numpy as np

class DQNAgent(BaseAgent):
    """
    Deep Q-Network Agent class.
    Args:
        boards_sample (ndarray): Sample board(s) used to determine the input shape of the model.
        alpha (float, optional): Learning rate. Defaults to 0.1.
        gamma (float, optional): Discount factor. Defaults to 0.95.
        epsilon (float, optional): Exploration factor. Defaults to 1.0.
        decay (float, optional): Decay rate for epsilon. Defaults to 0.99.
    Attributes:
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        epsilon (float): Exploration factor.
        decay (float): Decay rate for epsilon.
        input_shape (tuple): Shape of the input to the model.
        output_size (int): Number of possible actions.
        model (tensorflow.keras.Model): Deep Q-Network model.
    Methods:
        get_actions(boards, exploration=True):
            Returns the actions to take for the given boards.
        get_action(board):
            Returns the action to take for the given board.
        learn(prev_boards, actions, rewards, next_boards):
            Updates the Q-values of the model based on the given data.
        load_model_weights(path):
            Loads the model weights from the given path
    Private Methods:
        _board_to_input(boards):
            Converts the boards to the input format expected by the model.
        _build_model():
            Builds and compiles the Deep Q-Network model.
    """
    def __init__(self, boards_sample, alpha=0.1, gamma=0.95, epsilon=1.0, decay=0.99):
        """
        Initializes a DQNAgent object.

        Parameters:
        - boards_sample (ndarray): A sample of game boards used to determine the input shape of the model.
        - alpha (float): The learning rate for the agent's Q-learning algorithm. Default is 0.1.
        - gamma (float): The discount factor for future rewards in the agent's Q-learning algorithm. Default is 0.95.
        - epsilon (float): The exploration rate for the agent's epsilon-greedy policy. Default is 1.0.
        - decay (float): The decay rate for the agent's exploration rate. Default is 0.99.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay = decay
        # Model parameters
        if boards_sample.ndim == 2:
            boards_sample = boards_sample[np.newaxis, :, :]
        self.input_shape = self._board_to_input(boards_sample).shape[1:]
        self.output_size = 5  # UP, DOWN, LEFT, RIGHT, NONE
        self.model = self._build_model()

    def get_actions(self, boards, exploration=True):
        """
        Returns the actions to be taken by the agent based on the given game boards.
        Parameters:
        - boards (ndarray): The game boards for which actions need to be determined.
        - exploration (bool): Flag indicating whether to perform exploration or not.
        Returns:
        - actions (ndarray): The actions to be taken by the agent.
        """
        self.epsilon = self.epsilon * self.decay
        self.epsilon = max(self.epsilon, 0.01)
        if boards.ndim == 2:
            boards = boards[np.newaxis, :, :]
        input = self._board_to_input(boards)
        
        if np.random.rand() <= self.epsilon and exploration:
            return np.random.randint(0, self.output_size, size=(boards.shape[0], 1))
        else:
            q_values = self.model.predict(input)
            return [[action] for action in np.argmax(q_values, axis=1)]

    def get_action(self, board):
        """
        Get the action to take based on the given board state.
        Parameters:
        - board: numpy.ndarray
            The board state.
        Returns:
        - numpy.ndarray
            The action to take.
        """
        if board.ndim == 2:
            board = board[np.newaxis, :, :]
        n_boards = board.shape[0]
        input = board.reshape(n_boards, -1)
        
        q_values = self.model.predict(input)
        return np.argmax(q_values, axis=1)[:, np.newaxis]

    def learn(self, prev_boards, actions, rewards, next_boards):
        """
        Update the Q-values of the agent's model based on the observed rewards and next states.
        Args:
            prev_boards (list): List of previous game boards.
            actions (list): List of actions taken by the agent.
            rewards (list): List of rewards received by the agent.
            next_boards (list): List of next game boards.
        Returns:
            None
        """
        input = self._board_to_input(prev_boards)
        next_input = self._board_to_input(next_boards)
        
        current_q_values = self.model.predict(input)
        next_q_values = self.model.predict(next_input)
        
        max_next_q = np.max(next_q_values, axis=1)
        
        updated_q_values = current_q_values.copy()

        for i in range(len(rewards)):
            updated_q_values[i][actions[i][0]] = rewards[i][0] + self.gamma * max_next_q[i]
        
        self.model.fit(input, updated_q_values, epochs=1, verbose=0)
    
    def _board_to_input(self, boards):
        """
        Converts the given boards into categorical input for the DQN agent.

        Parameters:
        - boards: The input boards to be converted.

        Returns:
        - The categorical input for the DQN agent, with the first channel removed.

        """
        return K.utils.to_categorical(boards)[..., 1:]
    
    def _build_model(self):
        """
        Builds and compiles the deep Q-network model.

        Returns:
            model (tensorflow.keras.Model): The compiled deep Q-network model.
        """
        model = K.Sequential()
        model.add(K.layers.Flatten(input_shape=self.input_shape))
        model.add(K.layers.Dense(128, activation="relu"))
        model.add(K.layers.Dense(128, activation="relu"))
        model.add(K.layers.Dense(self.output_size, activation="linear"))
        model.compile(loss="mse", optimizer="adam")
        return model
    
    def load_model_weights(self, path):
        """
        Load the model weights from the given path.

        Parameters:
        - path (str): The path to the model weights file.

        Returns:
        - None
        """
        self.model.load_weights(path)