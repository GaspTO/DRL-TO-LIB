from math import sqrt, log

""" Select Greedy Evaluation Function for Tree Policies"""
class Evaluation_Strategy:
    def evaluate(self,node):
        pass


class UCT(Evaluation_Strategy):
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
        return self.exploration_weight * sqrt(self.log_N_vertex / (node.num_visits + 1))

    ''' exploitation '''
    def Q(self,node):
        return (-node.total_value) / (node.num_visits + 1)

    def evaluate(self,node):
        self.log_N_vertex = log(node.get_parent_node().num_visits + 1)
        U = self.U(node)
        Q = self.Q(node)
        return U + Q


class UCT_P(Evaluation_Strategy):
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
        return self.exploration_weight * node.policy_value * sqrt(self.log_N_vertex / (1 + node.num_visits))

    ''' exploitation '''
    def Q(self,node):
        return (-node.total_value) / (1 + node.num_visits)

    def evaluate(self,node):
        self.log_N_vertex = log(node.get_parent_node().num_visits + 1)
        U = self.U(node)
        Q = self.Q(node)
        return U + Q

class PUCT(Evaluation_Strategy):
    '''
    NEEDS:
        node.num_visits
        node.total_value
        node.policy_value
    '''
    def __init__(self,exploration_weight=1.0):
        self.exploration_weight = exploration_weight

    ''' exploration '''
    def U(self,node):
        return self.exploration_weight * node.policy_value * self.sqrt_N /(1 + node.num_visits)

    ''' exploitation '''
    def Q(self,node):
        return (-node.W)/(node.num_visits + 1)

    def evaluate(self,node):
        self.sqrt_N = sqrt(node.get_parent_node().num_visits)
        U = self.U(node)
        Q = self.Q(node)
        return U + Q



class SAVE_PUCT(Evaluation_Strategy):
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