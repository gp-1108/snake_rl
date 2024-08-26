<p align="center">
  <img src="https://img.icons8.com/?size=512&id=55494&format=png" width="20%" alt="<code>RL SNAKE Agents</code>-logo">
</p>
<p align="center">
    <h1 align="center"><code>RL Snake Agents</code></h1>
</p>
<p align="center">
    <em><code>MoE and DQL approaches to train RL agents for playing snake.</code></em>
</p>
<p align="center">
	<!-- Shields.io badges disabled, using skill icons. --></p>
<p align="center">
		<em>Built with the tools and technologies:</em>
</p>
<p align="center">
	<a href="https://skillicons.dev">
		<img src="https://skillicons.dev/icons?i=md,py,tensorflow">
	</a></p>

<br>

##### 🔗 Table of Contents

- [📍 Overview](#-overview)
- [📂 Repository Structure](#-repository-structure)
- [🧩 Modules](#-modules)
- [🚀 Getting Started](#-getting-started)
    - [🔖 Prerequisites](#-prerequisites)
    - [📦 Installation](#-installation)
    - [🤖 Usage](#-usage)

---

## 📍 Overview

This project showcases the development of a Reinforcement Learning (RL) agent capable of playing a modified version of the classic Snake game. The modifications include:

1. Collisions with walls result in a negative reward but do not reset the board.
2. When the snake eats itself, the portion of the body from the collision point to the tail is erased, with a negative reward issued, and the game continues.
3. Filling up the board grants a strong positive reward and resets the board.
4. The snake can legally stay in its current position.

### Agents Implemented

- **Random Agent**: A baseline agent that selects actions randomly, serving as a benchmark.
- **Baseline Agent**: A rule-based agent that prioritizes moving towards the fruit and, if stuck, triggers self-collision to clear space.
- **Deep Q-Learning Agent (DQL)**: A neural network-based agent that learns optimal strategies, especially in late-game scenarios.
- **Hybrid Agent**: Combines the strengths of the Baseline and DQL agents, using rule-based decisions in early game and DQL strategies in late game.

### Results

The Hybrid Agent outperformed others by leveraging the early-game efficiency of the Baseline Agent and the advanced strategic capabilities of the DQL Agent. It maximized fruit collection and minimized wall collisions, achieving superior performance on all benchmarks.

### Issues

While the Hybrid Agent excelled in the task-specific goals, it diverges from conventional Snake gameplay strategies, where self-collision typically ends the game. Adjusting the reward structure could lead to more traditional playstyles but would be outside the scope of the project.

### Conclusion

This project demonstrates the importance of setting appropriate rewards and goals in Reinforcement Learning, leading to innovative and effective strategies. The Hybrid Agent is a strong performer within the defined parameters, highlighting the potential of combining rule-based heuristics with deep learning techniques.

---

## 📂 Repository Structure

```sh
└── /
    ├── README.md
    ├── agents
    │   ├── BaseAgent.py
    │   ├── BaselineAgent.py
    │   ├── DQNAgent.py
    │   ├── HybridDQNAgent.py
    │   ├── RandomAgent.py
    │   ├── __init__.py
    ├── environments_fully_observable.py
    ├── evaluation.py
    ├── main.ipynb
    ├── requirements.txt
    ├── saved_models
    │   ├── dqn_model.weights.h5
    │   └── hybrid_model.weights.h5
    └── training.py
```

---

## 🧩 Modules

<details closed><summary>.</summary>

| File | Summary |
| --- | --- |
| [requirements.txt](requirements.txt) | Specifying dependencies in requirements.txt ensures that the project has the necessary libraries for optimal functionality and compatibility. It streamlines the setup process for contributors and users, fostering consistent environments for training and evaluation of agents within the repositorys architecture. |
| [environments_fully_observable.py](environments_fully_observable.py) | Establishes a foundational environment for snake-like agents in a simulated grid, facilitating board initialization, movement mechanics, and reward calculations. Integrates with reinforcement learning components, contributing to the overall architecture by providing necessary interactions for training and evaluation of various agent strategies. |
| [main.ipynb](main.ipynb) | This is the core of the project. It contains both training and evaluation of the agents. Have a look here for in depth info of the code. |
| [training.py](training.py) | This is a copy of esclusively the training part for the agents. You can run it to see the results by yourself and save perhaps an updated model version. |
| [evaluation.py](evaluation.py) | This is a copy of esclusively the evaluation part of the agents. You can run it to see the final results by yourself. |

</details>

<details closed><summary>agents</summary>

| File | Summary |
| --- | --- |
| [BaseAgent.py](agents/BaseAgent.py) |This code defines a `BaseAgent` class intended for game agents. It includes constants for game elements (e.g., snake's head, body, fruit) and movement directions. The class provides two methods, `get_actions` and `get_action`, which are meant to be overridden by subclasses to define agent behavior based on the game state. The class cannot be instantiated directly as the methods raise `NotImplementedError`. |
| [RandomAgent.py](agents/RandomAgent.py) | This code defines a `RandomAgent` class that inherits from `BaseAgent`. The `RandomAgent` randomly selects actions for a game. It has one attribute, `output_size`, representing the number of possible actions. The class includes two methods: `get_actions` to generate random actions for multiple game boards, and `get_action` to generate a random action for a single board. |
| [BaselineAgent.py](agents/BaselineAgent.py) | This code defines a `BaselineAgent` class that extends `BaseAgent` to make decisions based on the game board. The `BaselineAgent` calculates and returns optimal actions based on the proximity to fruit while avoiding walls and body parts. It uses a direction array to compute new head positions, clips them within board boundaries, and evaluates distances to the fruit. If all moves are illegal, it randomly selects a valid move. The class includes methods `get_actions` for multiple boards and `get_action` for a single board. |
| [DQNAgent.py](agents/DQNAgent.py) | This code defines a `DQNAgent` class implementing a Deep Q-Network (DQN) for reinforcement learning. The agent uses a neural network to predict Q-values for actions based on game board states. It includes attributes for learning rate (`alpha`), discount factor (`gamma`), exploration rate (`epsilon`), and a decay rate for epsilon. Key methods are: `get_actions()`: Selects actions for multiple boards using an epsilon-greedy strategy. `learn()`: Updates the Q-values based on rewards and new board states.
| [HybridDQNAgent.py](agents/HybridDQNAgent.py) | The `HybridDQNAgent` class combines baseline and DQN strategies. It uses the baseline approach for shorter body lengths and the DQN strategy for longer ones, avoiding wall collisions. It includes methods for selecting actions, adjusting for collisions, learning from experiences, and loading model weights. |

</details>

---

## 🚀 Getting Started

### 🔖 Prerequisites

**Python**: `version 3.12.4`

**Tensorflow**: `version 2.7.0`

**Numpy**: `version 1.21.2`

**Matplotlib**: `version 3.4.3`

### 📦 Installation

Build the project from source:

1. Clone the  repository:
```sh
❯ git clone .
```

2. Navigate to the project directory:
```sh
❯ cd snake_rl
```

3. Install the required dependencies:
```sh
❯ pip install -r requirements.txt
```

### 🤖 Usage

To run the project run the ```main.ipynb``` file. This file contains both the training and evaluation of the agents.

If you want to run the training and evaluation separately, you can run the ```training.py``` and ```evaluation.py``` files respectively.

```sh
❯ python training.py
❯ python evaluation.py
```