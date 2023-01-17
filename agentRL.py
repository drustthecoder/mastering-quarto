from quarto import Quarto, Player
import numpy as np
import random
import pickle

class AgentRL(Player):
    def __init__(self, game: Quarto, learn = False, alpha=0.2, random_factor=0.1):  # 80% explore, 20% exploit
        self.game = game
        self.state_history = []  # state, reward
        self.alpha = alpha
        self.random_factor = random_factor
        self.G = {}
        if learn:
            self.init_gains()
        else:
            with open('G.pkl', 'rb') as f:
                self.G = pickle.load(f)

    def init_gains(self):
        board_size = self.game.BOARD_SIDE
        possible_pieces = [-1] + list(range(16))
        for a in possible_pieces:
            for b in possible_pieces:
                for c in possible_pieces:
                    for d in possible_pieces:
                        for i in range(board_size):
                            self.G[(a, b, c, d, i)] = np.random.uniform(low=0.1, high=1)
    
    def get_free_places(self, board_status):
        free_places = np.where(board_status==-1)
        return [e for e in zip(free_places[1], free_places[0])]

    def get_row_column_diagonals_as_states(self, board_status, c, r):
        return (
            tuple(board_status[r]) + (r,),
            tuple(board_status[:, c]) + (c, ),
            tuple(board_status.diagonal().copy()) + (r, ),
            tuple(np.fliplr(board_status).diagonal().copy()) + (r, )
        )

    def place_piece(self) -> tuple[int, int]:
        maxG = -10e15
        randomN = np.random.random()
        if randomN < self.random_factor:
            # if random number below random factor, choose random action
            return random.randint(0, 3), random.randint(0, 3)
        else:
            # if exploiting, gather all possible actions and choose one with the highest G (reward)
            board_status = self.game.get_board_status()
            allowedMoves = self.get_free_places(board_status)
            for (c, r) in allowedMoves:
                board_data = self.get_row_column_diagonals_as_states(board_status, c, r)
                for state in board_data:
                    if self.G[state] >= maxG:
                        place_here = (c, r)
                        maxG = self.G[state]   
        return place_here
    
    def choose_piece(self) -> int:
        return random.randint(0, 15)

    def update_state_history(self, state, reward):
        self.state_history.append((state, reward))
    
    def reset_state_history(self):
        self.state_history = []

    def learn(self):
        target = 0
        self.state_history = list(reversed(self.state_history))
        for state, reward in self.state_history:
            self.G[state] = self.G[state] + self.alpha * (target - self.G[state])
            target += reward
        self.state_history = []
        self.random_factor -= 10e-5  # decrease random factor each episode of play