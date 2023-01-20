from quarto import Quarto, Player
import numpy as np
import random
import pickle
import copy
import datetime
import logging

class AgentRL(Player):
    def __init__(
        self, game: Quarto,
        learn_flag = False,
        endgame_num_of_pieces = 4,
        endgame_num_of_places=4,
        endgame_tree_search=False,
        time_limit=0,
        simulate_policy_in_MonteCarlo=True,
        alpha=0.2,
        random_factor=0.1):  # 80% explore, 20% exploit

        self.game = game
        self.endgame_num_of_pieces = endgame_num_of_pieces
        self.endgame_num_of_places = endgame_num_of_places
        self.endgame_tree_search = endgame_tree_search
        self.simulate_policy_in_MonteCarlo = simulate_policy_in_MonteCarlo
        self.time_limit = time_limit # microseconds
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

    def check_children(self, mygame: Quarto, place, piece, this_agent):
        # this_agent = mygame._Quarto__current_player
        # Play the game
        gameCopy = self.copy_game(mygame)
        gameCopy.select(piece)
        gameCopy._Quarto__current_player = 1 - mygame._Quarto__current_player
        gameCopy.place(*place)
        winner = gameCopy.check_winner()
        finished = gameCopy.check_finished()

        # If game finishes and there is no children
        if winner==this_agent:
            return 1
        elif winner==1-this_agent:
            return -1
        if finished:
            return 0

        # If game goes on
        board_status = gameCopy.get_board_status()
        free_pieces = self.get_free_pieces(board_status)
        free_places = self.get_free_places(board_status)
        value = 0
        for piece in free_pieces:
            for place in free_places:
                value += self.check_children(gameCopy, place, piece, this_agent)
        return value

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
    
    def copy_game(self, mygame: Quarto):
        gameCopy = Quarto()
        gameCopy._Quarto__board = copy.deepcopy(mygame._Quarto__board)
        gameCopy._Quarto__current_player = copy.deepcopy(mygame._Quarto__current_player)
        gameCopy._Quarto__selected_piece_index = copy.deepcopy(mygame._Quarto__selected_piece_index)
        return gameCopy

    def simulate_place_piece(self, mygame):
        # if self.random_place_piece:
        #     return random.randint(0, 3), random.randint(0, 3)
        # Try piece in every free place and if results in win, return the found place
        board_status = mygame.get_board_status()
        free_places = self.get_free_places(board_status)
        for place in free_places:
            gameCopy = self.copy_game(mygame)
            gameCopy.place(*place)
            winner = gameCopy.check_winner()
            if winner==gameCopy._Quarto__current_player:
                # we win
                return place
        # No piece results in win, return a random one
        return random.choice(free_places)

    def simulate_choose_piece(self, mygame: Quarto) -> int:
        if self.learn_flag:
            return random.randint(0, 15)
        board_status = mygame.get_board_status()
        free_pieces = self.get_free_pieces(board_status)
        free_places = self.get_free_places(board_status)
        score={} 
        for piece in free_pieces:
            score[piece] = 0
        pieces_with_zero_loss = []
        # Look in the future, Try each piece in every free place on board
        for piece in free_pieces:
            for place in free_places:
                gameCopy = self.copy_game(mygame)
                gameCopy.select(piece)
                gameCopy._Quarto__current_player = 1-gameCopy._Quarto__current_player
                gameCopy.place(*place)
                winner = gameCopy.check_winner()
                if winner==gameCopy._Quarto__current_player:
                    #opponent wins
                    score[piece]-=1
        max_score = -10e15
        for piece in free_pieces:
            if score[piece] >= max_score:
                selected_piece = piece
                max_score = score[piece]
        return selected_piece
            
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
        this_agent = self.game._Quarto__current_player
        logging.debug("Choose piece...")
        if self.learn_flag:
            return random.randint(0, 15)
        board_status = self.game.get_board_status()
        free_pieces = self.get_free_pieces(board_status)
        free_places = self.get_free_places(board_status)
        score={} 
        for piece in free_pieces:
            score[piece] = 0
        pieces_with_zero_loss = []
        # Look in the future, Try each piece in every free place on board
        for piece in free_pieces:
            for place in free_places:
                gameCopy = self.copy_game(self.game)
                gameCopy.select(piece)
                gameCopy._Quarto__current_player = 1-gameCopy._Quarto__current_player
                gameCopy.place(*place)
                winner = gameCopy.check_winner()
                if winner==gameCopy._Quarto__current_player:
                    #opponent wins
                    score[piece]-=1
            # if this piece does not result in loss in any place (never got a -1), return it without checking other pieces
            if score[piece]==0:
                pieces_with_zero_loss.append(piece)
        if len(pieces_with_zero_loss)==0:
            # If we're here, we only have bad pieces, choose the one which is the least bad
            max_score = -10e15
            for piece in free_pieces:
                if score[piece] >= max_score:
                    selected_piece = piece
                    max_score = score[piece]
            logging.debug("No winning piece! Returned the least bad piece!")
            return selected_piece
        elif len(pieces_with_zero_loss)==1:
            # we have one best piece, return it
            logging.debug("Returned the only no-loss piece!")
            return pieces_with_zero_loss[0]
        elif len(pieces_with_zero_loss)<self.endgame_num_of_pieces:
            if len(free_places)<self.endgame_num_of_places and self.endgame_tree_search:
                # Do a tree search
                t1 = datetime.datetime.now().timestamp()
                logging.debug(f"Doing tree search on places {free_places} and pieces {pieces_with_zero_loss}...")
                G = {}
                for piece in pieces_with_zero_loss:
                    for place in free_places:
                        G[piece] = self.check_children(self.game, place, piece, this_agent)
                logging.debug(dict(sorted(G.items(), key=lambda item: item[1], reverse=True)))
                max_score = -10e15
                for parent in G:
                    if G[parent] >= max_score:
                        selected_piece = parent # piece
                        max_score = G[parent]
                elapsed = (datetime.datetime.now().timestamp() - t1)*1000
                logging.debug(f"Elapsed {elapsed} microseconds, Selected {selected_piece}")
                return selected_piece
            if self.time_limit!=0:
                # Do simple Monte Carlo
                # Choose the best of the best
                logging.debug("Let's look into the future with a small amount of no-loss pieces!")
                G = {}
                for piece in pieces_with_zero_loss:
                    G[piece] = 0
                counter = 0
                loop_counter = 0
                first_loop_flag = True
                t1 = datetime.datetime.now().timestamp()
                elapsed = 0
                # While we have time, play the game until the game until the end for each current empty place on board as many times as possbile.
                while first_loop_flag or elapsed < self.time_limit:
                    loop_counter+=1
                    first_loop_flag = False
                    for piece in pieces_with_zero_loss:
                        if elapsed > self.time_limit:
                            logging.debug("Oops! Taking too much time to look into the future! Break!")
                            break
                        gameCopy = self.copy_game(self.game)
                        game_winner=-1
                        while game_winner<0 and not gameCopy.check_finished():
                            # player chooses piece
                            if self.simulate_policy_in_MonteCarlo:
                                gameCopy.select(self.simulate_choose_piece(gameCopy))
                            else:
                                game_free_pieces = self.get_free_pieces(gameCopy.get_board_status())
                                gameCopy.select(random.choice(game_free_pieces))
                            # player changes
                            gameCopy._Quarto__current_player = 1-gameCopy._Quarto__current_player
                            # player places the piece
                            if self.simulate_policy_in_MonteCarlo:
                                gameCopy.place(*self.simulate_place_piece(gameCopy))
                            else:
                                game_free_places = self.get_free_places(gameCopy.get_board_status())
                                gameCopy.place(*random.choice(game_free_places))
                            game_winner = gameCopy.check_winner()
                        # Give score to each place based on the result of the random play.
                        if game_winner==this_agent:
                            G[piece]+=1
                        # else:
                        #     G[piece]-=1
                        elapsed = (datetime.datetime.now().timestamp() - t1)*1000 # in microseconds, 1000 microseconds = 1 second
                    counter+=1
                    
                # on while exit
                maxG = -10e15
                for piece in pieces_with_zero_loss:
                    if G[piece]>=maxG:
                        maxG=G[piece]
                        selected_piece = piece
                logging.debug(f"elapsed {elapsed} microseconds, played {loop_counter} random games!")
                logging.debug(G)
                logging.debug(f"chose {selected_piece}")
                return selected_piece
        logging.debug("Too many pieces with zero loss, let's return a random one!")
        return random.choice(pieces_with_zero_loss)

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