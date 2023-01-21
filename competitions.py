from AgentShortSighted import AgentShortSighted
from AgentRL import AgentRL
from AgentMonteCarlo import AgentMonteCarlo
from AgentRandom import AgentRandom
from evaluate import evaluate
from quarto import Quarto
import logging


game = Quarto()

# game.reset()
# game.set_players((
#     AgentRL(game, choose_piece_enabled=False),
#     AgentShortSighted(game)))
# logging.basicConfig(level=logging.DEBUG)
# evaluate(game, 1000,
#     print_end_value="\r"
#     )

game.reset()
game.set_players((
    AgentRandom(game),
    AgentShortSighted(game)))
evaluate(game, 1000,
    print_end_value="\r"
    )