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

        choose_piece_time_limit = 3600,

        MonteCarlo_for_endgame_choose_piece_enabled = False,
        MonteCarlo_simulate_policy_enabled = False,
        MonteCarlo_endgame_num_of_pieces = 9,
        MonteCarlo_endgame_num_of_places = 9,

        tree_search_for_endgame_choose_piece_enabled = False,
        tree_search_endgame_num_of_pieces = 3,
        tree_search_endgame_num_of_places = 9,

        alpha=0.2,
        random_factor=0.1):  # 80% explore, 20% exploit

        self.game = game
        self.MonteCarlo_for_endgame_choose_piece_enabled = MonteCarlo_for_endgame_choose_piece_enabled
        self.MonteCarlo_simulate_policy_enabled = MonteCarlo_simulate_policy_enabled
        self.MonteCarlo_endgame_num_of_pieces = MonteCarlo_endgame_num_of_pieces
        self.MonteCarlo_endgame_num_of_places = MonteCarlo_endgame_num_of_places
        self.tree_search_endgame_num_of_pieces = tree_search_endgame_num_of_pieces
        self.tree_search_endgame_num_of_places = tree_search_endgame_num_of_places
        self.tree_search_for_endgame_choose_piece_enabled = tree_search_for_endgame_choose_piece_enabled
        self.choose_piece_time_limit = choose_piece_time_limit # microseconds
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
        # check all the children of the children
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
        # Look in the future, Try each piece in every free place on board
        if self.learn_flag:
            return random.randint(0, 15)
        board_status = mygame.get_board_status()
        free_pieces = self.get_free_pieces(board_status)
        free_places = self.get_free_places(board_status)
        score={} 
        for piece in free_pieces:
            score[piece] = 0
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
        sorted_score = sorted(score.items(), key=lambda item: item[1], reverse=True) 
        return sorted_score[0][0]
            
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
        # logging.debug("Choose piece...")
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
            # if this piece does not result in loss in any place (never got a -1), add it to pieces_with_zero_loss
            if score[piece]==0:
                pieces_with_zero_loss.append(piece)
        
        if len(pieces_with_zero_loss)==0:
            # If we're here, we only have bad pieces, choose the one which is the least bad
            sorted_score = sorted(score.items(), key=lambda item: item[1], reverse=True) 
            # logging.debug("No winning piece! Returned the least bad piece!")
            return sorted_score[0][0]
        elif len(pieces_with_zero_loss)==1:
            # we have one best piece, return it
            # logging.debug("Returned the only no-loss piece!")
            return pieces_with_zero_loss[0]
        if (self.tree_search_for_endgame_choose_piece_enabled and
            len(free_places)<self.tree_search_endgame_num_of_places and
            len(pieces_with_zero_loss)<self.tree_search_endgame_num_of_pieces):
            # Do a tree search
            return self.tree_search(pieces_with_zero_loss)
        if (self.MonteCarlo_for_endgame_choose_piece_enabled and
            self.choose_piece_time_limit != 0 and
            len(pieces_with_zero_loss) < self.MonteCarlo_endgame_num_of_pieces and
            len(free_places) < self.MonteCarlo_endgame_num_of_places):
            # Do a simple MonteCarlo
            return self.MonteCarlo(pieces_with_zero_loss)
        
        # random piece selection
        # logging.debug("Too many pieces with zero loss, let's return a random one!")
        return random.choice(pieces_with_zero_loss)

    def tree_search(self, pieces_with_zero_loss):
        # Do a tree search
        this_agent = self.game._Quarto__current_player
        board_status = self.game.get_board_status()
        free_places = self.get_free_places(board_status)
        t1 = datetime.datetime.now().timestamp()
        logging.debug(f"Doing tree search for {len(pieces_with_zero_loss)} pieces and {len(free_places)} places")
        G = {}
        t1 = datetime.datetime.now().timestamp()
        elapsed = 0
        for piece in pieces_with_zero_loss:
            for place in free_places:
                elapsed = (datetime.datetime.now().timestamp()-t1)*1000
                if elapsed>self.choose_piece_time_limit:
                    break
                G[piece] = self.check_children(self.game, place, piece, this_agent)
        sorted_G = sorted(G.items(), key=lambda item: item[1], reverse=True) 
        elapsed = (datetime.datetime.now().timestamp() - t1)*1000
        logging.debug(f"Elapsed {round(elapsed)} microseconds, Selected {sorted_G[0][0]}, sorted G (piece, score): {sorted_G}")
        return sorted_G[0][0]

    def MonteCarlo(self, pieces_with_zero_loss):
        # Do simple Monte Carlo
        this_agent = self.game._Quarto__current_player
        # Choose the best of the best
        logging.debug("Mone Carlo! Let's look into the future with all of no-loss pieces!")
        G = {}
        for piece in pieces_with_zero_loss:
            G[piece] = 0
        counter = 0
        loop_counter = 0
        first_loop_flag = True
        t1 = datetime.datetime.now().timestamp()
        elapsed = 0
        # While we have time, play the game until the game until the end for each current empty place on board as many times as possbile.
        while first_loop_flag or elapsed < self.choose_piece_time_limit:
            loop_counter+=1
            first_loop_flag = False
            for piece in pieces_with_zero_loss:
                if elapsed > self.choose_piece_time_limit:
                    break
                gameCopy = self.copy_game(self.game)
                game_winner=-1
                while game_winner<0 and not gameCopy.check_finished():
                    # player chooses piece
                    if self.MonteCarlo_simulate_policy_enabled:
                        gameCopy.select(self.simulate_choose_piece(gameCopy))
                    else:
                        game_free_pieces = self.get_free_pieces(gameCopy.get_board_status())
                        gameCopy.select(random.choice(game_free_pieces))
                    # player changes
                    gameCopy._Quarto__current_player = 1-gameCopy._Quarto__current_player
                    # player places the piece
                    if self.MonteCarlo_simulate_policy_enabled:
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
        sorted_G = sorted(G.items(), key=lambda item: item[1], reverse=True) 
        logging.debug(f"sorted (piece, score)[:5]: {sorted_G[:5]}")
        logging.debug(f"elapsed {round(elapsed)} microseconds, played {loop_counter} games for {len(sorted_G)} candidate piece, Selected {sorted_G[0][0]}")
        return sorted_G[0][0]

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