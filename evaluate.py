from quarto import Quarto
import numpy as np

def evaluate(sampleGame, cycles=50):
    players = sampleGame._Quarto__players
    # print("----------")
    # print(f"{type(players[0]).__name__} (goes first) vs. {type(players[1]).__name__}")
    agent1_win_count = 0
    agent2_win_count = 0
    tie_count = 0
    total_games = 0
    
    # for i in tqdm.tqdm(range(cycles)):
    print(f"{type(players[0]).__name__} vs. {type(players[1]).__name__}")
    for i in range(cycles):
        sampleGame.reset()
        winner = -1
        current_player = 0
        while winner < 0 and not sampleGame.check_finished():
            piece_ok = False
            piece = None
            while not piece_ok:
                # print(f"(P{current_player} ♟︎ )", end='')
                piece = players[current_player].choose_piece()
                piece_ok = sampleGame.select(piece)
            piece_ok = False
            current_player = 1 - current_player
            sampleGame._Quarto__current_player = current_player
            x, y = None, None
            while not piece_ok:
                # print(f"(P{current_player} 🙾 )", end='')
                x, y = players[current_player].place_piece()
                piece_ok = sampleGame.place(x, y)
            winner = sampleGame.check_winner()
        total_games += 1
        if winner==0:
            agent1_win_count+=1
            # print('(P1.WIN)', end="")
        elif winner==1:
            agent2_win_count+=1
            # print('(P2.WIN)', end="")
        else:
            tie_count+=1
            # print('(TIE)', end="")
        print(f"{type(players[0]).__name__} won: {round(agent1_win_count/total_games, 3)}, {type(players[1]).__name__} won: {round(agent2_win_count/total_games, 3)}, ties: {round(tie_count/total_games, 3)}, played {total_games} times!            ", end="\r")
    print("\n----------")
    # print(f"Played a total of {total_games} times!")