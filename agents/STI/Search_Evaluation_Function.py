from math import sqrt, log

""" Select Greedy Evaluation Function for Tree Policies"""
class Search_Evaluation_Function:
    def evaluate(self,node):
        pass


class UCT(Search_Evaluation_Function):
    '''
    NEEDS:
        node.num_chosen_by_parent
        node.num_losses
        node.num_wins
        node.num_draws
    '''
    def __init__(self,exploration_weight=1.0):
        self.exploration_weight = exploration_weight

    ''' exploration '''
    def U(self,node):
        return self.exploration_weight * sqrt(self.log_N_vertex / (1 + node.N))

    ''' exploitation '''
    def Q(self,node):
        return (-node.W) / (1 + node.N)

    def evaluate(self,node):
        self.log_N_vertex = log(node.get_parent_node().N)
        U = self.U(node)
        Q = self.Q(node)
        return U + Q

class PUCT(Search_Evaluation_Function):
    '''
    NEEDS:
        node.N
        node.W
        node.P
    '''
    def __init__(self,exploration_weight=1.0):
        self.exploration_weight = exploration_weight

    ''' exploration '''
    def U(self,node):
        return self.exploration_weight * node.P * self.sqrt_N /(1 + node.N)

    ''' exploitation '''
    def Q(self,node):
        return (-node.W)/(node.N + 1)

    def evaluate(self,node):
        self.sqrt_N = sqrt(node.get_parent_node().N)
        U = self.U(node)
        Q = self.Q(node)
        return U + Q



class SAVE_PUCT(Search_Evaluation_Function):
    '''
    NEEDS:
        node.num_chosen_py_parent
        node.num_losses
        node.num_draws
        node.num_wins
        node.p
    '''
    def __init__(self,exploration_weight=1.0):
        self.exploration_weight = exploration_weight

    def U(self,node):
        return self.exploration_weight * sqrt(self.log_N_parent/(node.num_chosen_by_parent+1))

    def Q(self,node):
        opponent_losses = node.num_losses + 0.5 * node.num_draws
        return (opponent_losses+node.p)/(node.num_chosen_by_parent + 1)

    def evaluate(self,node):
        self.log_N_parent = log(node.get_parent_node().num_chosen_by_parent)
        assert node.num_chosen_by_parent == node.num_losses + node.num_draws + node.num_wins
        U = self.U(node)
        Q = self.Q(node)
        return U + Q