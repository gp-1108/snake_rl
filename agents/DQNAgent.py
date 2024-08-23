from .BaseAgent import BaseAgent
import tensorflow.keras as K
import tensorflow as tf
import numpy as np

class DQNAgent(BaseAgent):
    def __init__(self, boards_sample, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_min=0, decay=0.99):
        self.alpha = alpha
        self.gamma = gamma
        self.lr = 1e-2
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay = decay
        self.temperature = 1.0 # For softmax
        # Model parameters
        if boards_sample.ndim == 2:
            boards_sample = boards_sample[np.newaxis, :, :]
        n_boards, height, width = boards_sample.shape
        self.input_size = height * width
        self.output_size = 5  # UP, DOWN, LEFT, RIGHT, NONE
        self.model = self._build_model()

    def get_actions(self, boards):
        if boards.ndim == 2:
            boards = boards[np.newaxis, :, :]
        n_boards = boards.shape[0]
        input = boards.reshape(boards.shape[0], -1)  # flatten the boards
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, self.output_size, size=(n_boards, 1))
        out = self.model.predict(input)
        # Apply softmax with temperature
        probs = tf.nn.softmax(out / self.temperature, axis=1)
        # Sample actions based on probabilities
        actions = tf.random.categorical(tf.math.log(probs), 1)
        return actions.numpy()

    def get_action(self, board):
        input = board.flatten()
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, self.output_size)
        out = self.model.predict(np.expand_dims(input, axis=0))
        # Apply softmax with temperature
        probs = tf.nn.softmax(out[0] / self.temperature)
        # Sample action based on probabilities
        action = tf.random.categorical(tf.math.log(probs.reshape(1, -1)), 1)
        return action.numpy()[0, 0]

    def learn(self, prev_boards, actions, rewards, next_boards):
        n_boards = prev_boards.shape[0]
        # Reshape previous boards
        input = prev_boards.reshape(n_boards, -1)
        # Predict target Q-values for current states
        target = self.model.predict(input)
        # Reshape next boards
        next_input = next_boards.reshape(n_boards, -1)
        # Predict target Q-values for next states
        next_target = self.model.predict(next_input)
        # Efficiently update target values using vectorized operations
        max_next_target = tf.reduce_max(next_target, axis=1)
        target[np.arange(n_boards), actions] = rewards + self.gamma * max_next_target
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