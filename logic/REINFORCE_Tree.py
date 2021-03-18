from agents.policy_gradient_agents.REINFORCE import REINFORCE, Config_Reinforce
import torch
import torch.nn as nn
from logic.gomoku_search import Basic_Gomoku_Search, MCTS_Gomoku_Search
from torch.distributions import Categorical
import random



class Config_Reinforce_Tree(Config_Reinforce):
    def __init__(self,config=None):
        Config_Reinforce.__init__(self,config)

    


class REINFORCE_Tree(REINFORCE):
    """ This is the one with one-look-ahead-BFS """
    def pick_action_and_get_log_probabilities(self):
        """Picks actions and then calculates the log probabilities of the actions it picked given the policy"""
        state = torch.from_numpy(self.get_state()).float().unsqueeze(0).to(self.device)
        action_values = self.policy(state,self.get_action_mask().unsqueeze(0))
        action_values_copy = action_values.detach()
        
        if(self.action_mask_required == True): #todo can't use the forward for this mask cause... critic_output
            mask = self.get_action_mask()
            unormed_action_values_copy =  action_values_copy.mul(mask)
            action_values_copy =  unormed_action_values_copy/unormed_action_values_copy.sum()
        
        action_distribution = Categorical(action_values_copy) # this creates a distribution to sample from
        
        search_result = Basic_Gomoku_Search(self.environment.state).BFS(1)
        if(search_result is not None):
            action = torch.tensor([search_result])
        else:
            action = action_distribution.sample()

        if(self.get_action_mask()[action]==0):
            print("fuck")
        
        if(self.debug_mode): self.logger.info("Q values\n {} -- Action chosen {} Masked_Prob {:.5f} True_Prob {:.5f}".format(action_values, action.item(),action_values_copy[0][action].item(),action_values[0][action].item()))
        else: self.logger.info("Action chosen {} Masked_Prob {:.5f} True_Prob {:.5f}".format(action.item(),action_values_copy[0][action].item(),action_values[0][action].item()))
        return action.item(), torch.log(action_values[0][action])



class REINFORCE_Tree_2(REINFORCE):
    """ This is the one with one-look-ahead-BFS """
    def pick_action_and_get_log_probabilities(self):
        """Picks actions and then calculates the log probabilities of the actions it picked given the policy"""
        state = torch.from_numpy(self.get_state()).float().unsqueeze(0).to(self.device)
        action_values = self.policy(state,self.get_action_mask().unsqueeze(0))
        action_values_copy = action_values.detach()
        
        if(self.action_mask_required == True): #todo can't use the forward for this mask cause... critic_output
            mask = self.get_action_mask()
            unormed_action_values_copy =  action_values_copy.mul(mask)
            action_values_copy =  unormed_action_values_copy/unormed_action_values_copy.sum()
        
        action_distribution = Categorical(action_values_copy) # this creates a distribution to sample from
        
        
        if(random.uniform(0,1) < 2):
            search_result = MCTS_Gomoku_Search(self.environment,self.environment.state).find_action_for_root(1000)
            action = torch.tensor([search_result])
            print("mcts : " + str(search_result))
        else:
            action = action_distribution.sample()
            print("normal action : " + str(action.item()))

        if(self.get_action_mask()[action]==0):
            print("fuck")
        
        if(self.debug_mode): self.logger.info("Q values\n {} -- Action chosen {} Masked_Prob {:.5f} True_Prob {:.5f}".format(action_values, action.item(),action_values_copy[0][action].item(),action_values[0][action].item()))
        else: self.logger.info("Action chosen {} Masked_Prob {:.5f} True_Prob {:.5f}".format(action.item(),action_values_copy[0][action].item(),action_values[0][action].item()))
        return action.item(), torch.log(action_values[0][action])
