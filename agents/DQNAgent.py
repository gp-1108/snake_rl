from .BaseAgent import BaseAgent
import tensorflow.keras as K
import tensorflow as tf
import numpy as np

class DQNAgent(BaseAgent):
    def __init__(self, boards, alpha=0.1, gamma=0.95):
        super().__init__(boards)
        self.alpha = alpha
        self.gamma = gamma
        self.lr = 1e-2

        # Model parameters
        self.n_boards, self.board_h, self.board_w = boards.shape
        self.input_size = self.board_h * self.board_w
        self.output_size = 5 # UP, DOWN, LEFT, RIGHT, NONE

        self.model = self._build_model()
    
    def get_actions(self):
        input = self.boards.reshape(self.n_boards, -1) # flatten the boards
        out = self.model.predict(input)
        best_actions = tf.argmax(out, axis=1)
        return np.expand_dims(best_actions, axis=1)
    
    def get_action(self, board):
        input = board.flatten()
        out = self.model.predict(input)
        best_action = tf.argmax(out)
        return best_action
    
    def learn(self, prev_boards, actions, rewards, next_boards):
        input = prev_boards.reshape(self.n_boards, -1)
        target = self.model.predict(input)
        next_input = next_boards.reshape(self.n_boards, -1)
        next_target = self.model.predict(next_input)
        for i in range(self.n_boards):
            target[i, actions[i]] = rewards[i] + self.gamma * tf.reduce_max(next_target[i])
        output = target + self.alpha * (target - target)
        self.model.train_on_batch(input, output)


    def _build_model(self):
        model = K.Sequential()
        model.add(K.layers.Dense(10, input_dim=self.input_size, activation="relu"))
        model.add(K.layers.Dense(10, activation="relu"))
        model.add(K.layers.Dense(self.output_size, activation="linear"))
        model.compile(loss="mse", optimizer=K.optimizers.Adam(learning_rate=self.lr))
        return model

