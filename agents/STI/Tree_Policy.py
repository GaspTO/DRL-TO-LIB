from agents.STI.Expansion_Strategy import Expansion_Strategy
from math import sqrt, log
import heapq
from agents.STI.Evaluation_Strategy import UCT, PUCT
from agents.STI.Search_Node import Search_Node
import numpy as np


""" Tree Policies - are the search policies in Search Iteration """
class Tree_Policy:
    def play(self,observation=None,debug=False):
        raise NotImplementedError


'''
GREEDY_DFS - for classic Monte Carlo Tree Search 
'''
class Greedy_DFS_Recursive(Tree_Policy):
    def __init__(self,environment,iterations,score_st,evaluation_st,expansion_st,backup_st):
        super().__init__()
        self.environment = environment
        self.iterations = iterations
        self.score_st = score_st
        self.evaluation_st = evaluation_st
        self.expansion_st = expansion_st
        self.backup_st = backup_st
        #* initialize root

    def play(self,observation=None,debug=False):
        if(observation is None): observation = self.environment.get_current_observation()
        self.root = Search_Node(self.environment,observation)
        self.expansion_st.initialize_node_attributes(self.root)
        for i in range(self.iterations):
            self._search(self.root,debug)
            print(self.root.W)
        #self._validate(self.root)
        return self._get_action_probabilities(self.root), {"root_node":self.root}
 
    def _search(self,node,debug=False):
        if not node.is_completely_expanded() and not node.is_terminal():
            self.expansion_st.expand(node)
        elif not node.is_terminal():
            next_node = max(node.get_successors(),key=self.evaluation_st.evaluate)
            self._search(next_node,debug)
        self.backup_st.update(node)
       
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
            if N != 1:
                raise ValueError("Visits in leaf is 1")


#!

    




'''
GREEDY_DFS - for classic Monte Carlo Tree Search 
'''
class Greedy_DFS(Tree_Policy):
    def __init__(self,evaluation_fn=None):
        super().__init__()
        if evaluation_fn is None:
            self.eval_fn = UCT()
        else:
            self.eval_fn = evaluation_fn

    def forward(self,debug=False):
        if debug: print("sel:",end="(")
        while True:
            if(self.node.get_parent_action() is None and debug): print("root",end=" ")
            elif debug: print(str(self.node.get_parent_action()),end=" ")
            if not self.node.is_completely_expanded() or self.node.is_terminal():
                return self.node
            self.node = max(self.node.get_successors(),key=self.eval_fn.evaluate)

    def reset(self,root):
        self.root = root
        self.node = root


'''
Ally decides over all possible adversary states
Adversary decides over local ally states
'''
class Adversarial_Greedy_Best_First_Search(Tree_Policy):
    def __init__(self,evaluation_fn=None):
        super().__init__()
        if evaluation_fn is None:
            self.eval_fn = UCT()
        else:
            self.eval_fn = evaluation_fn
    
    def forward(self):
        if len(self.greedy_frontier) == 0:
            self.node = self.root
        else:    
            self.node = heapq.heappop(self.greedy_frontier)[-1]
        while True:
            if not self.node.is_completely_expanded() or self.node.is_terminal():
                return self.node

            #* 1. I'm ally, put in adversary successors in greedy frontier, 
            if self.node.get_current_player() == self.root.get_current_player():
                for n in self.node.get_successors(): 
                    self.push_greedy_frontier(n)  
                self.node =  heapq.heappop(self.greedy_frontier)[-1]
                assert self.node.get_current_player() != self.root.get_current_player()

            #* 2. I'm adversary, choose best local ally
            elif self.node.get_current_player() != self.root.get_current_player():
                self.node = max(self.node.get_successors(),key=self.eval_fn.evaluate)
                assert self.node.get_current_player() == self.root.get_current_player()
               

    def push_greedy_frontier(self,node):
        heapq.heappush(self.greedy_frontier,(-1*self.eval_fn.evaluate(node),self.i,node)) #-1 cause piority is decreasing
        self.i += 1

    def reset(self,root):
        self.root = root
        self.i = 0
        self.greedy_frontier = []


'''
Every forward method call chooses the best node seen in heap and then
follows a greedy dfs until leaf
'''
class Local_Greedy_DFS_With_Global_Restart(Tree_Policy):
    def __init__(self,evaluation_fn=None):
        super().__init__()
        if evaluation_fn is None:
            self.eval_fn = UCT()
        else:
            self.eval_fn = evaluation_fn
    
    def forward(self):
        if len(self.greedy_frontier) == 0:
            self.node = self.root
        else:    
            self.node = heapq.heappop(self.greedy_frontier)[-1]
        while True:
            if not self.node.is_completely_expanded() or self.node.is_terminal():
                return self.node
            max_node = max(self.node.get_successors(),key=self.eval_fn.evaluate)
            #* 1. I'm ally, put in adversary successors in greedy frontier, 
            if self.node.get_current_player() == self.root.get_current_player():
                for n in self.node.get_successors():
                    if n != max_node:
                        self.push_greedy_frontier(n)  
            self.node = max_node
            
                
    def push_greedy_frontier(self,node,value = None):
        if value is None:
            heapq.heappush(self.greedy_frontier,(-1*self.eval_fn.evaluate(node),self.i,node)) #-1 cause piority is decreasing
        else:
            heapq.heappush(self.greedy_frontier,(value,self.i,node))
        self.i += 1

    def reset(self,root):
        self.root = root
        self.i = 0
        self.greedy_frontier = []




