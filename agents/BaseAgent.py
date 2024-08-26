class BaseAgent:
    """
    Base class for implementing game agents.
    Attributes:
        HEAD (int): Constant representing the head of the snake.
        BODY (int): Constant representing the body of the snake.
        FRUIT (int): Constant representing the fruit in the game.
        EMPTY (int): Constant representing an empty space in the game.
        WALL (int): Constant representing a wall in the game.
        UP (int): Constant representing the up direction.
        RIGHT (int): Constant representing the right direction.
        DOWN (int): Constant representing the down direction.
        LEFT (int): Constant representing the left direction.
        NONE (int): Constant representing no action.
    Methods:
        __init__(): Initializes a new instance of the BaseAgent class.
        get_actions(boards): Returns a list of possible actions for the given game boards.
        get_action(board): Returns the best action for the given game board.
    """
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

    def __init__(self):
        """
        Initializes the BaseAgent object.
        """
        raise NotImplementedError()
    
    def get_actions(self, boards):
        """
        Returns a list of actions based on the given game boards.

        Parameters:
        - boards: A np.array of shape (batch, height, width) of game boards representing the current state of the game.

        Returns:
        - A list of actions to be taken by the agent.

        Raises:
        - NotImplementedError: This method must be implemented by a subclass.
        """
        raise NotImplementedError()
    
    def get_action(self, board):
        """
        This method is used to get the action to be taken by the agent.

        Parameters:
        - board: The current state of the game board.

        Returns:
        - The action to be taken by the agent.

        Raises:
        - NotImplementedError: This method is meant to be overridden by subclasses.
        """
        raise NotImplementedError()