class BaseAgent:
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
        raise NotImplementedError()
    
    def get_actions(self, boards):
        raise NotImplementedError()
    
    def get_action(self, board):
        raise NotImplementedError()