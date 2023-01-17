from main import RandomPlayer
from quarto import Quarto
from agentRL import AgentRL
import pickle

if __name__ == '__main__':
    sampleGame = Quarto()
    agentRandom = RandomPlayer(sampleGame)
    agentRL = AgentRL(sampleGame, learn=True)
    players = (agentRL, agentRandom)
    sampleGame.set_players((agentRL, agentRandom))
    agentRL_win_count = 0
    agentRandom_win_count = 0
    tie_count = 0
    total_count = 1
    cycles = 20000
    for i in range(cycles):
        if i%100==0:
            print(f"{round(i/cycles*100, 1)}% completed! agentRL win ratio: {round(agentRL_win_count/total_count, 4)}")
        sampleGame.reset()
        winner = -1
        current_player = 1
        while winner < 0 and not sampleGame.check_finished():
            piece_ok = False
            piece = None
            while not piece_ok:
                piece = players[current_player].choose_piece()
                piece_ok = sampleGame.select(piece)
            board_state = sampleGame.get_board_status()
            piece_ok = False
            current_player = 1 - current_player
            sampleGame._Quarto__current_player = current_player
            c, r = None, None
            while not piece_ok:
                c, r = players[current_player].place_piece()
                piece_ok = sampleGame.place(c, r)
            winner = sampleGame.check_winner()
            if current_player==0:
                reward = 0 if winner==0 else -1
                board_status = sampleGame.get_board_status()
                board_data = agentRL.get_row_column_diagonals_as_states(board_status, c, r)
                for state in board_data:
                    agentRL.update_state_history(state, reward)
        if winner==0:
            agentRL.learn()
            agentRL_win_count+=1
        elif winner==1:
            agentRandom_win_count+=1
        else:
            tie_count+=1
        agentRL.reset_state_history()
        total_count+=1
    print(f"len(agentRL.G)={len(agentRL.G)}") 
    print(f"agentRL win count: {agentRL_win_count}, agentRandom win count: {agentRandom_win_count}, ties: {tie_count}!")
    with open('G.pkl', 'wb') as file:
        pickle.dump(agentRL.G, file)
         