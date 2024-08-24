from .BaseAgent import BaseAgent
import tensorflow.keras as K
import tensorflow as tf
import numpy as np

class DQNAgent(BaseAgent):
    def __init__(self, boards_sample, alpha=0.1, gamma=0.95, epsilon=1.0, decay=0.99):
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
        if board.ndim == 2:
            board = board[np.newaxis, :, :]
        n_boards = board.shape[0]
        input = board.reshape(n_boards, -1)
        
        q_values = self.model.predict(input)
        return np.argmax(q_values, axis=1)[:, np.newaxis]

    def learn(self, prev_boards, actions, rewards, next_boards):
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
        return K.utils.to_categorical(boards)[..., 1:]
    
    def _build_model(self):
        model = K.Sequential()
        model.add(K.layers.Flatten(input_shape=self.input_shape))
        model.add(K.layers.Dense(128, activation="relu"))
        model.add(K.layers.Dense(128, activation="relu"))
        model.add(K.layers.Dense(self.output_size, activation="linear"))
        model.compile(loss="mse", optimizer="adam")
        return model