from .BaseAgent import BaseAgent
import tensorflow.keras as K
import tensorflow as tf
import numpy as np

class DQNAgent(BaseAgent):
    def __init__(self, boards, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_min=0.05, decay=0.99):
        super().__init__(boards)
        self.alpha = alpha
        self.gamma = gamma
        self.lr = 1e-2
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay = decay
        # Model parameters
        self.n_boards, self.board_h, self.board_w = boards.shape
        self.input_size = self.board_h * self.board_w
        self.output_size = 5  # UP, DOWN, LEFT, RIGHT, NONE
        self.model = self._build_model()

    def get_actions(self):
        input = self.boards.reshape(self.n_boards, -1)  # flatten the boards
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, self.output_size, size=(self.n_boards, 1))
        out = self.model.predict(input)
        best_actions = tf.argmax(out, axis=1)
        return np.expand_dims(best_actions, axis=1)

    def get_action(self, board):
        input = board.flatten()
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, self.output_size)
        out = self.model.predict(np.expand_dims(input, axis=0))
        best_action = tf.argmax(out[0])
        return best_action.numpy()

    def learn(self, prev_boards, actions, rewards, next_boards):
        # Reshape previous boards
        input = prev_boards.reshape(self.n_boards, -1)
        # Predict target Q-values for current states
        target = self.model.predict(input)
        # Reshape next boards
        next_input = next_boards.reshape(self.n_boards, -1)
        # Predict target Q-values for next states
        next_target = self.model.predict(next_input)
        # Efficiently update target values using vectorized operations
        max_next_target = tf.reduce_max(next_target, axis=1)
        target[np.arange(self.n_boards), actions] = rewards + self.gamma * max_next_target
        # Fit the model to the updated targets
        self.model.fit(input, target, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.decay
        else:
            self.epsilon = self.epsilon_min

    def _build_model(self):
        model = K.Sequential()
        model.add(K.layers.Dense(64, input_dim=self.input_size, activation="relu"))
        model.add(K.layers.Dense(64, activation="relu"))
        model.add(K.layers.Dense(self.output_size, activation="linear"))
        model.compile(loss="mse", optimizer=K.optimizers.Adam(learning_rate=self.lr))
        return model