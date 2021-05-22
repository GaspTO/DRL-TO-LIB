from agents.search_agents.minimax.Minimax import Minimax
from agents.search_agents.minimax.Minimax_Value_Estimation_Strategy import *
import torch


''' To implement ... '''


def minimax_expert(self,observation,max_depth):
    env = self.environment.environment

    #* value estimation policy
    #value_estimation = Random_Rollout_Estimation(num_rollouts=1)
    #value_estimation = Network_Value_Estimation(self.network,self.device)
    value_estimation = Network_Q_Estimation(self.network,self.device)

    #* tree policy
    tree_policy = Minimax(env,value_estimation,max_depth=max_depth)
    

    action_probs, info = tree_policy.play(observation)
    root_node = info["root_node"]

    #!sampling
    #action_distribution = Categorical(torch.tensor(action_probs)) # this creates a distribution to sample from
    #action = action_distribution.sample()
    action = action_probs.argmax()

    return action, torch.FloatTensor(action_probs), torch.tensor([root_node.value])