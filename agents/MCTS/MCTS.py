from agents.MCTS.Expansion_Strategy import Expansion_Strategy
from agents.Agent import Agent
from agents.MCTS.Evaluation_Strategy import UCT, PUCT
from agents.MCTS.MCTS_Search_Node import MCTS_Search_Node
import numpy as np





'''
Normal MCTS - with recursive calls

Only works for 1 or 2 players
'''
class MCTS(Agent):
    def __init__(self,environment,iterations,score_st,evaluation_st,expansion_st):
        Agent.__init__(self,environment)
        self.environment = environment
        self.iterations = iterations
        self.score_st = score_st
        self.evaluation_st = evaluation_st
        self.expansion_st = expansion_st
        #* initialize root

    def play(self,observation=None,debug=False):
        if(observation is None): observation = self.environment.get_current_observation()
        self.root = MCTS_Search_Node(self.environment,observation)
        self.expansion_st.expand(self.root)
        for i in range(self.iterations):
            self._search(self.root,debug)
        if debug: self._validate(self.root)
        return self._get_action_probabilities(self.root), {"root_node":self.root}

    def _search(self,node,debug=False):
        if node.is_terminal():
            self._backtrack(node,0)
        elif not node.is_completely_expanded():
            leaf_value = self.expansion_st.expand(node)
            self._backtrack(node,leaf_value)
        else: 
            next_node = max(node.get_successors(),key=self.evaluation_st.evaluate)
            self._search(next_node,debug)
       
    def _backtrack(self,node,leaf_value):
        path_value = leaf_value #leaf is in node.get_player() prespective
        while node is not None:
            if node.get_parent_node() is not None:
                if node.get_parent_node().get_player() == node.get_player():
                    path_value = path_value + node.get_parent_reward()
                    node.total_value += path_value
                else:
                    path_value = path_value + -1*node.get_parent_reward()
                    node.total_value += path_value
                    path_value = -1* path_value #path value will change prespective for parent
            else:
                node.total_value += path_value
            node.num_visits += 1
            node = node.get_parent_node()

    def _get_action_probabilities(self,node):
        #the length of successors is not always the action_size 'cause invalid actions don't become successors
        action_probs = np.zeros(self.environment.get_action_size()) 
        for n in self.root.get_successors():
            action_probs[n.parent_action] = self.score_st.summarize(n)
        # if a vector is full of zeros 
        if(action_probs.sum() == 0.):
            for n in self.root.get_successors():
                action_probs[n.parent_action] = 1/len(self.root.get_successors()) 
        else:
            action_probs = action_probs/action_probs.sum()
            if len(np.where(action_probs < 0)[0]) != 0:
                raise ValueError("Negative probability in vector")
        return action_probs
        
    ''' traverses the tree to debug '''
    def _validate(self,node):
        print(node.depth)
        N = node.N
        W = node.W
        for n in node.get_successors():
            N -= n.N
            W += n.W 
            self._validate(n)
        if len(node.get_successors()) != 0:
            if (node == self.root and N != 0) or (node != self.root and N != 1):
                raise ValueError("Number of Root Visits should equal")
            if (node == self.root and W != 0):
                raise ValueError("Number of W should equal")
        else:
            if N != 1 and N != 0 and not node.is_terminal():
                raise ValueError("Visits in leaf is not 1")

    
