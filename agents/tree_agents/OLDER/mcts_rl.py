from agents.Agent import Agent
from agents.tree_agents.Node import Gomoku_MCTSNode, K_Row_MCTSNode, MCTS_FIRST_PLAYER, MCTS_TIE, MCTS_SECOND_PLAYER, MCTS_WIN, MCTS_LOSS
from agents.tree_agents.Searchfuck import MCTS_Search

class mcts_rl_agent(Agent):
	def __init__(self,n,environment):
		self.n = n
		self.environment = environment

	def play(self,state):
		search = MCTS_Search(K_Row_MCTSNode(self.environment.environment.state))
		search.run_n_playouts(self.n)
		action = search.play_action()
		return action
