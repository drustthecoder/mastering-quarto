from quarto import Quarto, Player
import numpy as np
import random
import pickle
import copy

class AgentRL(Player):
    def __init__(self, game: Quarto, learn_flag = False, alpha=0.2, random_factor=0.1):  # 80% explore, 20% exploit
        self.game = game
        self.state_history = []  # state, reward
        self.alpha = alpha
        self.random_factor = random_factor
        self.G = {}
        self.learn_flag = learn_flag
        if learn_flag:
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

    def get_free_pieces(self, board_status):
        played_pieces = [e for e in board_status[board_status!=-1]]
        return list(set(range(16))-set(played_pieces))
    
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
    
    def copy_game(self):
        gameCopy = Quarto()
        gameCopy._Quarto__board = copy.deepcopy(self.game._Quarto__board)
        gameCopy._Quarto__current_player = copy.deepcopy(self.game._Quarto__current_player)
        gameCopy._Quarto__selected_piece_index = copy.deepcopy(self.game._Quarto__selected_piece_index)
        return gameCopy

    def place_piece(self) -> tuple[int, int]:
        # return random.randint(0, 3), random.randint(0, 3)
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
        if self.learn_flag:
            return random.randint(0, 15)
        board_status = self.game.get_board_status()
        free_pieces = self.get_free_pieces(board_status)
        free_places = self.get_free_places(board_status)
        G={} 
        for piece in free_pieces:
            G[piece] = 0
        # Look in the future, Try each piece in every free place on board
        for piece in free_pieces:
            for place in free_places:
                gameCopy = self.copy_game()
                gameCopy.select(piece)
                gameCopy._Quarto__current_player = 1-gameCopy._Quarto__current_player
                gameCopy.place(*place)
                winner = gameCopy.check_winner()
                if winner==gameCopy._Quarto__current_player:
                    #opponent wins
                    G[piece]-=1
            # if this piece does not result in loss in any place (never got a -1), return it without checking other pieces
            if G[piece]==0:
                return piece
        # If we're here, we only have bad pieces, choose the one which is the least bad
        maxG = -10e15
        for piece in free_pieces:
            if G[piece] >= maxG:
                selected_piece = piece
                maxG = G[piece]
        return selected_piece

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