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

    def evaluate(self,node):
        log_N_vertex = log(node.get_parent_node().num_chosen_by_parent)
        assert node.num_chosen_by_parent == node.num_losses + node.num_draws + node.num_wins
        opponent_losses = node.num_losses + 0.5 * node.num_draws
        return opponent_losses / node.num_chosen_by_parent + self.exploration_weight * sqrt(log_N_vertex / (node.num_chosen_by_parent))


class PUCT(Search_Evaluation_Function):
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

    def evaluate(self,node):
        sqrt_N = sqrt(node.get_parent_node().num_chosen_by_parent)
        assert node.num_chosen_by_parent == node.num_losses + node.num_draws + node.num_wins
        opponent_losses = node.num_losses + 0.5 * node.num_draws
        U = self.exploration_weight * node.p * sqrt_N /(1 + node.num_chosen_by_parent)
        Q = opponent_losses/(node.num_chosen_by_parent + 1)
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

    def evaluate(self,node):
        log_N_parent= log(node.get_parent_node().num_chosen_by_parent)
        assert node.num_chosen_by_parent == node.num_losses + node.num_draws + node.num_wins
        opponent_losses = node.num_losses + 0.5 * node.num_draws
        U = self.exploration_weight * sqrt(log_N_parent/(node.num_chosen_by_parent+1))
        Q = (opponent_losses+node.p)/(node.num_chosen_by_parent + 1)
        return U + Q