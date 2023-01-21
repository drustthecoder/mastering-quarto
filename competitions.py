from AgentShortSighted import AgentShortSighted
from AgentRL import AgentRL
from AgentMonteCarlo import AgentMonteCarlo
from AgentRandom import AgentRandom
from evaluate import evaluate
from quarto import Quarto
import logging


game = Quarto()

game.reset()
game.set_players((
    AgentRL(game),
    AgentRandom(game)))
logging.basicConfig(level=logging.DEBUG)
evaluate(game, 100,
    print_end_value="\r"
    )