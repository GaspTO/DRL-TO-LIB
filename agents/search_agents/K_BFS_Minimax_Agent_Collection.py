from agents.search_agents.k_best_first_minimax.K_Best_First_Minimax import K_Best_First_Minimax
from agents.search_agents.k_best_first_minimax.K_Best_First_Minimax_Expansion_Strategy import *
from agents.search_agents.Abstract_Network_Search_Agent import Abstract_Network_Search_Agent


""" Just an agent simplification for the different stategies """


class K_Best_First_Minimax_Rollout(K_Best_First_Minimax):
    def __init__(self,environment,k,num_iterations,num_rollouts_per_node,debug=False):
        expansion_st = K_Best_First_All_Successors_Rollout_Expansion_Strategy(num_rollouts=num_rollouts_per_node)
        K_Best_First_Minimax.__init__(self,environment,expansion_st,k=k,num_iterations=num_iterations,debug=debug)


class K_Best_First_Minimax_Q(K_Best_First_Minimax,Abstract_Network_Search_Agent):
    def __init__(self,environment,k,num_iterations,network):
        expansion_st = K_Best_First_Network_Successor_Q_Expansion_Strategy(network)
        K_Best_First_Minimax.__init__(self,environment,expansion_st,k=k,num_iterations=num_iterations)
        Abstract_Network_Search_Agent.__init__(self,network)
    
    def set_network(self,network):
        Abstract_Network_Search_Agent.set_network(self,network)

        
class K_Best_First_Minimax_V(K_Best_First_Minimax,Abstract_Network_Search_Agent):
    def __init__(self,environment,k,num_iterations,network):
        expansion_st = K_Best_First_Network_Successor_V_Expansion_Strategy(network)
        K_Best_First_Minimax.__init__(self,environment,expansion_st,k=k,num_iterations=num_iterations)




