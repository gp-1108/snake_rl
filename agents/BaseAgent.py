class BaseAgent:
    def __init__(self, boards):
        self.boards = boards
    
    def get_actions(self):
        raise NotImplementedError()
    
    def get_action(self, board):
        raise NotImplementedError()

    def learn(self, actions, rewards):
        raise NotImplementedError()