from agents.Agent import Agent
from agents.search_agents.best_first_minimax.Best_First_Minimax_Node import Best_First_Search_Node
import torch
import numpy as np

class Best_First_Minimax(Agent):
    def __init__(self,environment,expansion_st,num_iterations=None):
        Agent.__init__(self,environment)
        self.expansion_st = expansion_st
        self.num_iterations = num_iterations

    def play(self,observation=None,debug=False):
        if(observation is None): observation = self.environment.get_current_observation()
        self.root = Best_First_Search_Node(self.environment,observation)
        self._minimax_search(self.root,self.num_iterations,debug)
        return self._get_action_probabilities(self.root), {"root_node":self.root}

    def _minimax_search(self,root,num_iterations,debug=False):
        for i in range(num_iterations):
            frontier_node = self._find_best_unexpanded_node(root)
            if frontier_node == None:
                return
            else:
                self.expansion_st.expand(frontier_node)
                self._backtrack(frontier_node)
            pass
                
    def _find_best_unexpanded_node(self,node):
        queue = []
        while node.is_completely_expanded() or node.is_terminal():
            if not node.is_terminal():
                ordered_successors = node.get_successors()
                ordered_successors.sort(key=self._get_parent_estimation_though_successor)
                queue.extend(ordered_successors)   
            if len(queue) == 0:
                return None
            else:
                node = queue.pop()
        return node

    def _backtrack(self,node):
        if node is not None:
            best_succ_node = max(node.get_successors(),key=self._get_parent_estimation_though_successor)
            node.value = self._get_parent_estimation_though_successor(best_succ_node)
            self._backtrack(node.get_parent_node())
            
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
        
        assert not action_probs.sum().item() > 1
            
        return action_probs
        
