from environments.core.Players import Players, Player, IN_GAME, TERMINAL, TIE_PLAYER_NUMBER
import torch
import random

""" Expand """
class Expansion_Strategy:
    def __init__(self):
        pass

    def update_tree(self,node):
        pass

    def initialize_node_attributes(self,node):
        pass

    def reset(self):
        pass

'''
THIS IS FROM THE ORIGINAL MCTS
WHEN update_tree IS CALLED, A RANDOM SUCCESSOR OF IT IS GENERATED
THEN A RANDOM ROLLOUT FROM IT IS USED TO ESTIMATE ITS VALUE
'''
class One_Successor_Rollout(Expansion_Strategy):
    '''
        Uses node.N, node.W
    '''
    def __init__(self,win_points=0,loss_points=-1,draw_points=-0.5):
        self.win_points = win_points
        self.loss_points = loss_points
        self.draw_points = draw_points
        super().__init__()

    def initialize_node_attributes(self,node):
        node.N = 0
        node.W = 0

    def update_tree(self,node):
        if not node.is_terminal():
            succ_node = self._expand_new(node)
            winner_player = self._simulate_from_random_node(succ_node)
            print("[" + str(succ_node.get_parent_action()) + "=" + str(winner_player==succ_node.get_current_player())+ "])",end="")
            self._backpropagate(succ_node, winner_player)
        else:
            winner_player = node.get_winner_player()
            self._backpropagate(node,winner_player)

    def _expand_new(self,node):
        #* expands random node
        succ_node = node.expand_random_successor()
        succ_node.belongs_to_tree = True
        self.initialize_node_attributes(succ_node)
        return succ_node
        

    def _simulate_from_random_node(self,succ_node):
        #* estimation: random rolllout
        rollout_node = succ_node
        while not rollout_node.is_terminal():
            rollout_node = rollout_node.find_random_unexpanded_successor()
            rollout_node.belongs_to_tree = False
        winner_player = rollout_node.get_winner_player()
        return winner_player

    def _backpropagate(self,node,winner,debug=False):
        while not node.is_root():
            self._update_statistic(node,winner)
            node = node.get_parent_node()
        assert node.is_root()

        '''debug'''
        if(node.get_current_player() == winner and debug): print(" win",end=" ")
        elif TIE_PLAYER_NUMBER == winner.get_number() and debug:
            print("draw")
        elif debug: print(" lose",end=" ")

        self._update_statistic(node,winner)

    def _update_statistic(self,node,winner):
            if node.belongs_to_tree:
                node.N += 1
                if TIE_PLAYER_NUMBER == winner.get_number():
                    node.W += self.draw_points
                elif node.get_current_player().get_number() == winner.get_number():
                    node.W += self.win_points
                else:
                    node.W += self.loss_points





"""
THIS IS THE MCTS RL
"""
class Network_One_Successor_Rollout(One_Successor_Rollout):
    '''
        Uses node.N, node.W, node.P
    '''
    def __init__(self,network,device):
        super().__init__()
        self.network = network
        self.device = device

    def update_tree(self,node):
        if not node.is_terminal():
            succ_node = self._expand_new(node)
            winner_player = self._simulate_from_random_node(succ_node)
            self._backpropagate(succ_node, winner_player)
        else:
            winner_player = node.get_winner_player()
            self._backpropagate(node,winner_player)

    def initialize_node_attributes(self,node):
        node.N = 0
        node.W = 0
        node.P = 0

    def _expand_new(self,node):
        ''' expand all and adds node.P '''
        nodes = node.expand_rest_successors()
        current_board = node.get_current_observation()
        x = torch.from_numpy(current_board).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            self.network.load_state(x)
            policy_values = self.network.get_policy_values(apply_softmax=True, mask=torch.tensor(node.get_mask()))
        for node_child in nodes:
            self.initialize_node_attributes(node_child)
            node_child.P = policy_values[0][node_child.parent_action].item()
            node_child.belongs_to_tree = True
        random_idx = random.randint(0,len(nodes)-1)
        return nodes[random_idx]



class Network_Policy_Value(Expansion_Strategy):
    '''
        Uses node.N, node.W, node.P
    '''
    def __init__(self,network,device):
        super().__init__()
        self.network = network
        self.device = device

    def update_tree(self,node):
        if not node.is_terminal():
            leaf_node, leaf_player, leaf_points = self._expand(node)
            self._backpropagate(leaf_node, leaf_player,leaf_points)
        else:
            leaf_node, leaf_player, leaf_points  = self._evaluate_terminal_state(node)
            self._backpropagate(leaf_node, leaf_player,leaf_points)

    def initialize_node_attributes(self,node):
        node.N = 0
        node.W = 0
        node.P = 0

    def _expand(self,node):
        ''' expand all and adds node.p '''
        if node.is_terminal():
            raise ValueError("Should not be terminal") 
        succ_nodes = node.expand_rest_successors()
        current_board = node.get_current_observation()
        x = torch.from_numpy(current_board).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            self.network.load_state(x)
            policy_values = self.network.get_policy_values(apply_softmax=True, mask=torch.tensor(node.get_mask()))
            state_estimate = self.network.get_state_value()
            assert node.W == 0
            node.W = state_estimate #* create node W
        for node_child in succ_nodes:
            self.initialize_node_attributes(node_child)
            node_child.P = policy_values[0][node_child.parent_action].item()
        return node, node.get_current_player(), state_estimate

    def _evaluate_terminal_state(self,node):
        if not node.is_terminal():
            raise ValueError("Should be terminal") 

        winner = node.get_winner_player()
        #! rewards are hardcoded
        if TIE_PLAYER_NUMBER == winner.get_number():
            points = -0.5 #! debug 0
        elif node.get_current_player().get_number() == winner.get_number():
            points = 0 #! 1
        else:
            points = -1
        return node,node.get_current_player(),points

    def _backpropagate(self,node,leaf_player,leaf_points):
        while not node.is_root():
            self._update_statistic(node,leaf_player,leaf_points)
            node = node.get_parent_node()
        self._update_statistic(node,leaf_player,leaf_points)

    def _update_statistic(self,node,player,points):
        node.N += 1
        if node.get_current_player() == player:
            node.W += points
        else:
            node.W -= points


        

class Normal_With_Network_Estimation(Expansion_Strategy):
    '''
        Uses node.N, node.W, node.P
    '''
    def __init__(self,network,device):
        super().__init__()
        self.network = network
        self.device = device

    def update_tree(self,node):
        if not node.is_terminal():
            leaf_node, leaf_player, leaf_points = self._expand(node)
            self._backpropagate(leaf_node, leaf_player,leaf_points)
        else:
            leaf_node, leaf_player, leaf_points  = self._evaluate_terminal_state(node)
            self._backpropagate(leaf_node, leaf_player,leaf_points)

    def initialize_node_attributes(self,node):
        node.N = 0
        node.W = 0

    def _expand(self,node):
        ''' expand all and adds node.p '''
        if node.is_terminal():
            raise ValueError("Should not be terminal") 
        succ_nodes = node.expand_rest_successors()
        current_board = node.get_current_observation()
        x = torch.from_numpy(current_board).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            self.network.load_state(x)
            state_estimate = self.network.get_state_value()
            node.W = state_estimate #* create node W
        for node_child in succ_nodes:
            self.initialize_node_attributes(node_child)
        return node, node.get_current_player(), state_estimate

    def _evaluate_terminal_state(self,node):
        if not node.is_terminal():
            raise ValueError("Should be terminal") 

        winner = node.get_winner_player()
        #! rewards are hardcoded
        if TIE_PLAYER_NUMBER == winner.get_number():
            points = 0 #! debug should be 0
        elif node.get_current_player().get_number() == winner.get_number():
            points = 1 #! debug should be 1
        else:
            points = -1#....
        return node,node.get_current_player(),points

    def _backpropagate(self,node,leaf_player,leaf_points):
        while not node.is_root():
            self._update_statistic(node,leaf_player,leaf_points)
            node = node.get_parent_node()
        self._update_statistic(node,leaf_player,leaf_points)

    def _update_statistic(self,node,player,points):
        node.N += 1
        if node.get_current_player() == player:
            node.W += points
        else:
            node.W -= points