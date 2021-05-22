from agents.search_agents.mcts.MCTS import MCTS
from agents.search_agents.mcts.MCTS_Evaluation_Strategy import *
from agents.search_agents.mcts.MCTS_Expansion_Strategy import *

"""
I HAVEN'T FINISHED THIS FILE
THE GOAL IS TO DO RELEVANT SEARCH AGENTS AND ABSTRACT HOW THEY'RE CREATED LIKE K_BFS_Minimax_Agent_Collection
"""

'''
def mcts_original(self,observation,n,exploration_weight):
    #todo some things in here need config
    search = MCTS_Agents.MCTS_Search(self.environment.environment,n,exploration_weight=exploration_weight)
    action = search.play(observation)
    return action

def mcts_simple_rl(self,observation,n,exploration_weight):
    #todo some things in here need config
    search = MCTS_Agents.MCTS_Simple_RL_Agent(self.environment.environment,n,self.network,self.device,exploration_weight=exploration_weight)
    action = search.play(observation)
    probs = search.probs
    return action,probs
'''

''' New Search agents '''
def mcts_expert(self,observation,iterations,exploration_weight=1.0):
    env = self.environment.environment

    #* eval functions
    tree_evaluation = UCT(exploration_weight=exploration_weight)
    #tree_evaluation = UCT_P(exploration_weight=1.0)
    #tree_evaluation = PUCT(exploration_weight=1.0)

    #* expand policy
    #! expand policy
    #tree_expansion = MCTS_One_Successor_Rollout()
    tree_expansion = MCTS_Network_Value(self.network,self.device)
    #tree_expansion = MCTS_Network_Policy_Value(self.network,self.device)
    #tree_expansion = MCTS_Network_Policy_One_Successor_Rollout(self.network,self.device)


    #* tree policy 
    tree_policy = MCTS(env,iterations,evaluation_st=tree_evaluation,expansion_st=tree_expansion)
    
    #!
    action_probs, info = tree_policy.play(observation)
    root_node = info["root_node"]

    action = action_probs.argmax()

    return action, torch.FloatTensor(action_probs), torch.tensor([root_node.total_value/root_node.num_visits])