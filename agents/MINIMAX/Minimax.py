from agents.Agent import Agent
from agents.MINIMAX.Minimax_Search_Node import Minimax_Search_Node
import numpy as np
import torch


class Minimax(Agent):
    def __init__(self,environment,value_estimation,max_depth=None):
        Agent.__init__(self,environment)
        self.value_estimation = value_estimation
        self.max_depth = max_depth

    def play(self,observation=None,debug=False):
        if(observation is None): observation = self.environment.get_current_observation()
        self.root = Minimax_Search_Node(self.environment,observation)
        self._minimax_search(self.root,self.max_depth,debug)
        return self._get_action_probabilities(self.root), {"root_node":self.root}

    def _minimax_search(self,node,depth,debug=False):
        if depth == 0 or node.is_terminal():
            node.value = self.value_estimation.estimate(node)
        else:
            successors = node.expand_rest_successors()
            current_value = float("-inf")
            for n in successors:
                self._minimax_search(n,depth-1,debug)
                n.value = n.get_parent_reward() + -1 * n.value
                if current_value < n.value:
                    current_value = n.value
            node.value = current_value
    
    def _estimate_node(self,node,num_rollouts):
        accumulated_reward = 0.
        for i in range(num_rollouts):
            rollout_node = node
            while not rollout_node.is_terminal():
                parent_player = rollout_node.get_player()
                rollout_node = rollout_node.find_random_successor()
                if node.get_player() == parent_player:
                    accumulated_reward += rollout_node.get_parent_reward()
                else:
                    accumulated_reward += -1*rollout_node.get_parent_reward()
        return accumulated_reward/num_rollouts

    def _get_action_probabilities(self,node):
        #the length of successors is not always the action_size 'cause invalid actions don't become successors
        action_probs = torch.zeros(self.environment.get_action_size()) 
        mask = torch.tensor(self.environment.get_mask())
        action_probs = torch.where(mask == 0,torch.tensor(-1e18),action_probs)
        for n in node.get_successors():
            assert action_probs[n.get_parent_action()] == 0.
            action_probs[n.get_parent_action()] = n.value
        action_probs = torch.softmax(action_probs,dim=0).numpy()
        return action_probs
        

        
            
  

        
        

        
