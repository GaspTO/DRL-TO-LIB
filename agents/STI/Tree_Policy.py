from agents.STI.Expansion_Strategy import Expansion_Strategy
from math import sqrt, log
import heapq
from agents.STI.Evaluation_Strategy import UCT, PUCT
from agents.STI.Search_Node import Search_Node
import numpy as np


"""

'''
MINIMAX - The classic minimax
'''
class Minimax(Agent):
    def __init__(self,environment,depth,score_st,evaluation_st,expansion_st,backup_st):
        super().__init__()
        self.environment = environment
        self.depth = depth
        self.score_st = score_st
        self.evaluation_st = evaluation_st
        self.expansion_st = expansion_st
        self.backup_st = backup_st
        #* initialize root


    def _search(self,node,debug=False):
        if not node.is_completely_expanded():
            self.expansion_st.expand(node)

        if node.is_terminal():
            self.backup_st.update(node)
        
        

        for n in node.get_successors():
            value = max(value, self._search(n,debug))


        if not node.is_completely_expanded() and not node.is_terminal():
            self.expansion_st.expand(node)
        elif not node.is_terminal():
            for n in node.get_successors():
                self._search(n,debug)
        self.backup_st.update(node)



function minimax(node, depth, maximizingPlayer) is
    if depth = 0 or node is a terminal node then
        return the heuristic value of node
    if maximizingPlayer then
        value := −∞
        for each child of node do
            value := max(value, minimax(child, depth − 1, FALSE))
        return value
    else (* minimizing player *)
        value := +∞
        for each child of node do
            value := min(value, minimax(child, depth − 1, TRUE))
        return value






#! BELOW IS OUTDATED



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




"""