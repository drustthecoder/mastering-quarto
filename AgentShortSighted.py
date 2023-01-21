from quarto import Quarto, Player
import numpy as np
import random
import copy

class AgentShortSighted(Player):
    """
    This agent uses a kind of tree search for piece selection and piece placement.
    (Breadth-first, limit-depth=1)
    It checks all the nodes in the next layer of the game tree.
    For piece placement, if any move results in win, it returns that move.
    for piece selection
      - If the piece results in a loss it lowers the piece score. after checking all available pieces in all available places, it checks for the piece with the highest score and returns it.
      - If the piece doesnt not result in loss in any place, returns it immediately without checking other pieces.
    For even better results agains more powerful agents:
    - If the piece placement doesn't result in an instant win, what kind of placement can lower the chance of loss or increase the chance of win in the next play
    - if the piece selection doesn't result in an instant loss, which piece can lower the chance of chance of loss or increase the chance of win in the next play
    """
    def __init__(self, game: Quarto, random_place_piece = False, random_choose_piece = False):
        self.game = game
        self.random_place_piece = random_place_piece
        self.random_choose_piece = random_choose_piece
    
    def copy_game(self):
        gameCopy = Quarto()
        gameCopy._Quarto__board = copy.deepcopy(self.game._Quarto__board)
        gameCopy._Quarto__current_player = copy.deepcopy(self.game._Quarto__current_player)
        gameCopy._Quarto__selected_piece_index = copy.deepcopy(self.game._Quarto__selected_piece_index)
        return gameCopy

    def get_free_pieces(self, board_status):
        played_pieces = [e for e in board_status[board_status!=-1]]
        return list(set(range(16))-set(played_pieces))

    def get_free_places(self, board_status):
        free_places = np.where(board_status==-1)
        return [e for e in zip(free_places[1], free_places[0])]

    def choose_piece(self):
        if self.random_choose_piece:
            return random.randint(0, 15)
        board_status = self.game.get_board_status()
        free_pieces = self.get_free_pieces(board_status)
        free_places = self.get_free_places(board_status)
        G={} 

        if (len(free_pieces)>8 or len(free_places)>8):
            return random.randint(0, 15)

        for piece in free_pieces:
            G[piece] = 0
        # Look in the future, Try each piece in every free place on board
        for piece in free_pieces:
            for place in free_places:
                gameCopy = self.copy_game()
                gameCopy.select(piece)
                gameCopy._Quarto__current_playe = 1-gameCopy._Quarto__current_player
                gameCopy.place(place[0], place[1])
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

    def place_piece(self):
        if self.random_place_piece:
            return random.randint(0, 3), random.randint(0, 3)
        # Try piece in every free place and if results in win, return the found place
        board_status = self.game.get_board_status()
        free_places = self.get_free_places(board_status)
        for place in free_places:
            gameCopy = self.copy_game()
            gameCopy.place(*place)
            winner = gameCopy.check_winner()
            if winner==gameCopy._Quarto__current_player:
                # we win
                return place
        # No piece results in win, return a random one
        return random.choice(free_places)