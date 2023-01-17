from quarto import Quarto, Player
import numpy as np
import random
import copy
import datetime

class AgentMonteCarlo(Player):
    """
    This agent tries to look one step ahead in the future for piece selection and piece placement.
    It used the same base algorithm as the winCheck agent.
    However, when the base algorithm doesn't result in as instant win in the piece placement, it tries random sampling for free place on the board and chooses the one with the highest score.
    simulate_policy parameter control the agent policy for the random sampling:
    - If simulate_policy is true, the agent uses its own piece selection and a simplified version of its own piece placement policy in the future lookups
    - If simulate_policy is false, the agent uses random choice instead of its own policy for the future lookup
    """
    def __init__(self, game: Quarto, time_limit=1, simulate_policy=True):
        self.game = game
        # How long should it take (in seconds) for place_piece() to return the answer
        self.time_limit = time_limit
        # When looking in the future, should we simulate our own policy when it's our turn or should we use random steps
        self.simulate_policy = simulate_policy
    
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
        t1 = datetime.datetime.now()
        this_agent = self.game._Quarto__current_player
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
        # No place results in an instant win. We are going to do random sampling.
        G = {}
        for place in free_places:
            G[place] = 0
        elapsed = datetime.datetime.now() - t1
        counter = 0
        # While we have time, play the game until the game until the end for each current empty place on board as many times as possbile.
        while elapsed.seconds < self.time_limit:
            for place in free_places:
                gameCopy = self.copy_game()
                gameCopy.place(*place)
                # we don't check for win because we know that no piece results in instant win
                game_winner=-1
                while game_winner<0 and not gameCopy.check_finished():
                    # player chooses piece
                    board_status = gameCopy.get_board_status()
                    game_free_pieces = self.get_free_pieces(board_status)
                    gameCopy.select(random.choice(game_free_pieces))
                    # player changes
                    gameCopy._Quarto__current_player = 1-gameCopy._Quarto__current_player
                    # player places the piece
                    board_status = gameCopy.get_board_status()
                    game_free_places = self.get_free_places(board_status)
                    gameCopy.place(*random.choice(game_free_places))
                    game_winner = gameCopy.check_winner()
                # Give score to each place based on the result of the random play.
                if game_winner==this_agent:
                    G[place]+=1
                else:
                    G[place]-=1
            counter+=1
            elapsed = datetime.datetime.now() - t1
        # Choose the place that has the highest score
        maxG = -10e15
        for place in free_places:
            if G[place]>=maxG:
                maxG=G[place]
                place_here = place
        return place_here
    