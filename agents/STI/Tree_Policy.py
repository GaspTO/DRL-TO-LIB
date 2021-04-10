from math import sqrt, log
import heapq

""" Select Greedy Evaluation Function for Tree Policies"""
class Search_Evaluation_Function:
    def evaluate(self,node):
        pass

class UCT(Search_Evaluation_Function):
    def __init__(self,exploration_weight=1.0):
        self.exploration_weight = exploration_weight

    def evaluate(self,node):
        log_N_vertex = log(node.get_parent_node().num_chosen_by_parent)
        assert node.num_chosen_by_parent == node.num_losses + node.num_draws + node.num_wins
        opponent_losses = node.num_losses + 0.5 * node.num_draws
        return opponent_losses / node.num_chosen_by_parent + self.exploration_weight * sqrt(log_N_vertex / node.num_chosen_by_parent)


class PUCT(Search_Evaluation_Function):
    '''
    needs node to have a probability given by neural network in node.p
    '''
    def __init__(self,exploration_weight=1.0):
        self.exploration_weight = exploration_weight

    def evaluate(self,node):
        sqrt_N = sqrt(node.num_chosen_by_parent)
        assert node.num_chosen_by_parent == node.num_losses + node.num_draws + node.num_wins
        opponent_losses = node.num_losses + 0.5 * node.num_draws
        U = self.exploration_weight * node.p * sqrt_N /(1 + node.num_chosen_by_parent)
        Q = opponent_losses/(node.num_chosen_by_parent + 1)
        return U + Q
 


""" Tree Policies  """
class Tree_Policy:
    """
    needs to use self.frontier
    """
    def __init__(self,search):
        self.search = search

    def forward(self):
        pass

    def reset(self):
        pass



class Greedy_DFS(Tree_Policy):
    def __init__(self,search,evaluation_fn=None):
        super().__init__(search)
        if evaluation_fn is None:
            self.eval_fn = UCT()

    def forward(self):
        if self.search.debug: print("SEL:",end='')
        while True:
            if self.search.debug: print(str(self.search.current_node.get_parent_action()),end=" ")
            if not self.search.current_node.is_completely_expanded() or self.search.current_node.is_terminal():
                if self.search.debug: print("\n")
                return
            self.frontier.append(max(self.search.current_node.get_successors(),key=self.eval_fn.evaluate))
            self.search.current_node = self.frontier.pop()

    def reset(self):
        self.frontier = []


'''
Ally decides over all possible adversary states
Adversary decides over local ally states
'''
class Adversarial_Greedy_Best_First_Search(Tree_Policy):
    def __init__(self,search,evaluation_fn=None):
        super().__init__(search)
        if evaluation_fn is None:
            self.eval_fn = UCT()
    
    def forward(self):
        if self.search.debug: print("SEL:",end='')
        while True:
            if not self.search.current_node.is_completely_expanded() or self.search.current_node.is_terminal():
                if self.search.debug: print("\n")
                return

            #* 1. I'm ally, put in adversary successors in greedy frontier, 
            if self.search.current_node.get_current_player() == self.search.root.get_current_player():
                for n in self.search.current_node.get_successors(): 
                    self.push_greedy_frontier(n)  
                self.search.current_node =  heapq.heappop(self.greedy_frontier)[-1]
                if self.search.debug: print(str(self.search.current_node.get_parent_action()),end=" ")
                assert self.search.current_node.get_current_player() != self.search.root.get_current_player()

            #* 2. I'm adversary, choose best local ally
            elif self.search.current_node.get_current_player() != self.search.root.get_current_player():
                self.search.current_node = max(self.search.current_node.get_successors(),key=self.eval_fn.evaluate)
                if self.search.debug: print(str(self.search.current_node.get_parent_action()),end=" ")
                assert self.search.current_node.get_current_player() == self.search.root.get_current_player()
        if self.search.debug: print("")                

    def push_greedy_frontier(self,node):
        heapq.heappush(self.greedy_frontier,(-1*self.eval_fn.evaluate(node),self.i,node)) #-1 cause piority is decreasing
        self.i += 1

    def reset(self):
        self.i = 0
        self.greedy_frontier = []




