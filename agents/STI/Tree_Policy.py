from math import sqrt, log
import heapq
from agents.STI.Search_Evaluation_Function import UCT, PUCT

 


""" Tree Policies  """
class Tree_Policy:
    """
    needs to use self.frontier
    """
    def __init__(self):
        pass

    def forward(self):
        pass

    def reset(self):
        pass



class Greedy_DFS(Tree_Policy):
    def __init__(self,evaluation_fn=None):
        super().__init__()
        if evaluation_fn is None:
            self.eval_fn = UCT()
        else:
            self.eval_fn = evaluation_fn

    def forward(self):
        while True:
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
        while True:
            if not self.node.is_completely_expanded() or self.node.is_terminal():
                if self.search.debug: print("\n")
                return self.node

            #* 1. I'm ally, put in adversary successors in greedy frontier, 
            if self.node.get_current_player() == self.root.get_current_player():
                for n in self.node.get_successors(): 
                    self.push_greedy_frontier(n)  
                self.node =  heapq.heappop(self.greedy_frontier)[-1]
                assert self.node.get_current_player() != self.root.get_current_player()

            #* 2. I'm adversary, choose best local ally
            elif self.node.get_current_player() != self.root.get_current_player():
                self.node = max(self.snode.get_successors(),key=self.eval_fn.evaluate)
                assert self.node.get_current_player() == self.root.get_current_player()
               

    def push_greedy_frontier(self,node):
        heapq.heappush(self.greedy_frontier,(-1*self.eval_fn.evaluate(node),self.i,node)) #-1 cause piority is decreasing
        self.i += 1

    def reset(self,root):
        self.root = root
        self.node = self.root
        self.i = 0
        self.greedy_frontier = []




