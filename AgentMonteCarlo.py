import numpy as np
from quarto import Player, Quarto
import random
from main import RandomPlayer
import copy
import datetime

class AgentMonteCarlo(Player):
    def __init__(self, game: Quarto, time_limit=250, simulate_policy=True):
        self.game = game
        # How long should it take (in seconds) for place_piece() to return the answer
        self.time_limit = time_limit
        # When looking in the future, should we simulate our own policy when it's our turn or should we use random steps
        self.simulate_policy = simulate_policy
    
    def copy_game(self, original_game, this_agent):
        gameCopy = Quarto()
        gameCopy._Quarto__board = copy.deepcopy(original_game._Quarto__board)
        gameCopy._Quarto__current_player = copy.deepcopy(original_game._Quarto__current_player)
        gameCopy._Quarto__selected_piece_index = copy.deepcopy(original_game._Quarto__selected_piece_index)
        gameCopy._Quarto__players = [original_game._Quarto__players[this_agent] for i in range(2)]
        return gameCopy

    def get_free_pieces(self, board_status):
        played_pieces = [e for e in board_status[board_status!=-1]]
        return list(set(range(16))-set(played_pieces))

    def get_free_places(self, board_status):
        free_places = np.where(board_status==-1)
        return [e for e in zip(free_places[1], free_places[0])]

    def simulate_choose_piece(self, mygame):
        free_pieces = self.get_free_pieces(mygame.get_board_status())
        free_places = self.get_free_places(mygame.get_board_status())
        G={} 
        for piece in free_pieces:
            G[piece] = 0
        # Look in the future, Try each piece in every free place on board
        for piece in free_pieces:
            for place in free_places:
                gameCopy = self.copy_game(mygame, mygame._Quarto__current_player)
                gameCopy.select(piece)
                gameCopy._Quarto__current_player = 1 - gameCopy._Quarto__current_player
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

    def choose_piece(self):
        free_pieces = self.get_free_pieces(self.game.get_board_status())
        free_places = self.get_free_places(self.game.get_board_status())
        G={} 
        for piece in free_pieces:
            G[piece] = 0
        # Look in the future, Try each piece in every free place on board
        for piece in free_pieces:
            for place in free_places:
                gameCopy = self.copy_game(self.game, 1 - self.game._Quarto__current_player)
                gameCopy.select(piece)
                gameCopy._Quarto__current_player = 1 - gameCopy._Quarto__current_player
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

    def simulate_place_piece(self, mygame):
        # Try piece in every free place and if results in win, return the found place
        free_places = self.get_free_places(mygame.get_board_status())
        for place in free_places:
            gameCopy = self.copy_game(mygame, mygame._Quarto__current_player)
            gameCopy.place(*place)
            winner = gameCopy.check_winner()
            if winner==gameCopy._Quarto__current_player:
                # we win
                return place
        # No piece results in win, return a random one
        return random.choice(free_places)

    def place_piece(self):
        
        this_agent = self.game._Quarto__current_player
        # Try piece in every free place and if results in win, return the found place
        free_places = self.get_free_places(self.game.get_board_status())
        for place in free_places:
            gameCopy = self.copy_game(self.game, self.game._Quarto__current_player)
            gameCopy.place(*place)
            winner = gameCopy.check_winner()
            if winner==gameCopy._Quarto__current_player:
                # we win
                return place
        # No place results in an instant win. We are going to do random sampling.
        G = {}
        for place in free_places:
            G[place] = 0
        # elapsed = datetime.datetime.now() - t1
        counter = 0
        # While we have time, play the game until the game until the end for each current empty place on board as many times as possbile.
        loop_counter = 0
        first_loop_flag = True
        t1 = datetime.datetime.now().timestamp()
        while first_loop_flag or elapsed < self.time_limit:
            loop_counter+=1
            first_loop_flag = False
            for place in free_places:
                gameCopy = self.copy_game(self.game, this_agent)
                gameCopy.place(*place)
                # we don't check for win because we know that no piece results in instant win
                game_winner=-1
                while game_winner<0 and not gameCopy.check_finished():
                    # player chooses piece
                    if self.simulate_policy and gameCopy._Quarto__current_player==this_agent:
                        gameCopy.select(self.simulate_choose_piece(gameCopy))
                    else:
                        game_free_pieces = self.get_free_pieces(gameCopy.get_board_status())
                        gameCopy.select(random.choice(game_free_pieces))
                    # player changes
                    gameCopy._Quarto__current_player = 1-gameCopy._Quarto__current_player
                    # player places the piece
                    if self.simulate_policy and gameCopy.get_player()==this_agent:
                        gameCopy.place(*self.simulate_place_piece(gameCopy))
                    else:
                        game_free_places = self.get_free_places(gameCopy.get_board_status())
                        gameCopy.place(*random.choice(game_free_places))
                    game_winner = gameCopy.check_winner()
                # Give score to each place based on the result of the random play.
                if game_winner==this_agent:
                    G[place]+=1
                else:
                    G[place]-=1
            counter+=1
            elapsed = (datetime.datetime.now().timestamp() - t1)*1000 # in microseconds, 1000 microseconds = 1 second
            # print(elapsed)
        # Choose the place that has the highest score
        # print(loop_counter)
        maxG = -10e15
        for place in free_places:
            if G[place]>=maxG:
                maxG=G[place]
                place_here = place
        return place_here
    