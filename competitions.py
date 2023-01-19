from AgentShortSighted import AgentShortSighted
from AgentRL import AgentRL
from AgentMonteCarlo import AgentMonteCarlo
from AgentRandom import AgentRandom
from evaluate import evaluate
from quarto import Quarto
import logging

game = Quarto()

# game.reset()
# game.set_players((AgentShortSighted(game, random_place_piece=True), AgentRandom(game)))
# evaluate(game, 100)

game.reset()
game.set_players((AgentRL(game, time_limit=500), AgentRandom(game)))
# logging.basicConfig(level=logging.DEBUG)
evaluate(game, 100)

# game.reset()
# game.set_players((AgentRL(game, time_limit=1000), AgentRandom(game)))
# evaluate(game, 1000)

# game.reset()
# game.set_players((AgentShortSighted(game), AgentRL(game)))
# evaluate(game, 50)

# game.reset()
# game.set_players((AgentMonteCarlo(game), AgentRL(game)))
# evaluate(game, 50)