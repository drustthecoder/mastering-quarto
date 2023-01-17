from shortSighted import AgentShortSighted
from agentRL import AgentRL
from main import RandomPlayer
from customClasses import evaluate
from quarto.objects import Quarto
from MonteCarlo import AgentMonteCarlo

game = Quarto()

# game.reset()
# game.set_players((AgentRL(game), RandomPlayer(game)))
# evaluate(game, 1000)

# game.reset()
# game.set_players((AgentRL(game), AgentShortSighted(game)))
# evaluate(game, 1000)

game.reset()
game.set_players((AgentShortSighted(game), RandomPlayer(game)))
evaluate(game, 1000)

# game.reset()
# game.set_players((AgentMonteCarlo(game), AgentRL(game)))
# evaluate(game, 10)

# game.reset()
# game.set_players((AgentMonteCarlo(game), AgentShortSighted(game)))
# evaluate(game, 10)

# game.reset()
# game.set_players((AgentShortSighted(game), AgentShortSighted(game)))
# evaluate(game, 1000)