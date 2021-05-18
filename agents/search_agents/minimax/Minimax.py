from agents.Agent import Agent
from agents.search_agents.minimax.Minimax_Node import Minimax_Node
import numpy as np
import torch


class Minimax(Agent):
    def __init__(self,environment,value_estimation_st,max_depth=None):
        Agent.__init__(self,environment)
        self.value_estimation_st = value_estimation_st
        self.max_depth = max_depth

    def play(self,observation=None,debug=False):
        if(observation is None): observation = self.environment.get_current_observation()
        self.root = Minimax_Node(self.environment,observation)
        self._minimax_search(self.root,self.max_depth,debug)
        return self._get_action_probabilities(self.root), {"root_node":self.root}

    def _minimax_search(self,node,depth,debug=False):
        if node.is_terminal():
            node.value = 0
        elif depth == 0:
            node.value = self.value_estimation_st.estimate_node(node)
        else:
            successors = node.expand_rest_successors()
            current_value = float("-inf")
            for n in successors:
                self._minimax_search(n,depth-1,debug)
                succ_estimation = self._get_parent_estimation_though_successor(n)
                if current_value < succ_estimation:
                    current_value = succ_estimation
            node.value = current_value

    def _get_parent_estimation_though_successor(self,succ):
        if succ.get_player() == succ.get_parent_node().get_player():
            return succ.get_parent_reward() + succ.value #edge between n and succ plus rest of path estimation
        else:
            return succ.get_parent_reward() + -1*succ.value
    
    def _get_action_probabilities(self,node):
        #the length of successors is not always the action_size 'cause invalid actions don't become successors
        action_probs = np.zeros(self.environment.get_action_size()) 
        mask = self.environment.get_mask()
        num_max_values = 0
        for n in node.get_successors():
            assert action_probs[n.get_parent_action()] == 0.
            succ_estimation = self._get_parent_estimation_though_successor(n)
            if succ_estimation == self.root.value:
                num_max_values += 1
                action_probs[n.get_parent_action()] = 1
        action_probs = action_probs * (1/num_max_values)
        
        if action_probs.sum().item() > 1:
            print("ups")
        return action_probs
        

