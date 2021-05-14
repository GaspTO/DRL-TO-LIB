from agents.Agent import Agent
from agents.search_agents.best_first_search.Best_First_Search_Node import Best_First_Search_Node
import torch
import numpy as np

class Best_First_Minimax(Agent):
    def __init__(self,environment,value_estimation,num_expansions=None):
        Agent.__init__(self,environment)
        self.value_estimation = value_estimation
        self.num_expansions = num_expansions

    def play(self,observation=None,debug=False):
        if(observation is None): observation = self.environment.get_current_observation()
        self.root = Best_First_Search_Node(self.environment,observation)
        self.root.N = 0
        self._minimax_search(self.root,self.num_expansions,debug)
        return self._get_action_probabilities(self.root), {"root_node":self.root}

    def _minimax_search(self,root,num_expansions,debug=False):
        for i in range(num_expansions):
            frontier_node = self._find_best_unexpanded_node(root)
            if frontier_node == None:
                return
            else:
                successors = frontier_node.expand_rest_successors()
                for n in successors:
                    n.value = self.value_estimation.estimate_node(n)
                    n.N = 0
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
        ''' if node.is_completely_expanded() and not node.is_terminal():
                if node.get_player() == node.get_parent_node().get_player():
                    node_future_reward_estimation_fn = lambda n: n.get_parent_reward() + n.value #edge between n and succ plus rest of path estimation
                else:
                    node_future_reward_estimation_fn = lambda n: n.get_parent_reward() + -1*n.value
        '''
        if node is not None:
            best_succ_node = max(node.get_successors(),key=self._get_parent_estimation_though_successor)
            node.value = self._get_parent_estimation_though_successor(best_succ_node)
            self._backtrack(node.get_parent_node())
            node.N += 1
            
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
        
        #minimum = action_probs.min()
        #action_probs = (action_probs - minimum)
        #action_probs = np.where(mask == 0,0.,action_probs)
        #action_probs = action_probs/(action_probs.sum())
       # action_probs2 = np.where(action_probs == self.root.value,1/num_max_values,0.)
        if action_probs.sum().item() > 1:
            print("ups")
        return action_probs
        
