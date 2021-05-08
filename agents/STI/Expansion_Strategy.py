from environments.core.Players import Players, Player, IN_GAME, TERMINAL, TIE_PLAYER_NUMBER
import torch
import random

""" Expand """
class Expansion_Strategy:
    def __init__(self):
        pass

    ''' expand one or more successors of node
    and updates node's attributes like number of visits '''
    def expand(self,node):
        pass

    ''' this method will be used externally to initialize root '''
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
    def __init__(self):
        super().__init__()

    def initialize_node_attributes(self,node):
        node.N = 0
        node.W = 0 #* total delta_subtree_R + reward from parent to node - in prespective of node
        node.delta_subtree_R = 0 #*value of all subtrees in prespective from node
        #* needs get_parent_reward()

    ''' Expands current node
    it has to add visits and W to random successor chosen
    since that is invisible to the search '''
    def expand(self,node):
        if not node.is_terminal():
            successor = self._choose_random_successor(node)
            successor.delta_subtree_R += self._estimate_node(successor)
            self._backup_successor_node(successor)
        else:
            raise ValueError("shouldn't be terminal")

    def _choose_random_successor(self,node):
        #* expands random node
        succ_node = node.expand_random_successor()
        self.initialize_node_attributes(succ_node)
        return succ_node
        
    #* returns the value of subtree total reward from the prespective of succ_node
    def _estimate_node(self,succ_node):
        def get_rollout_reward_from_succ_prespective(succ_node,rollout_node):
            if rollout_node.get_player() != succ_node.get_parent_node().get_player():
                return -1 * rollout_node.get_parent_reward()
            else:
                 return rollout_node.get_parent_reward()

        #* estimation: random rolllout
        rollout_node = succ_node
        delta_subtree_R = 0.
        while not rollout_node.is_terminal():
            rollout_node = rollout_node.find_random_unexpanded_successor()
            delta_subtree_R += get_rollout_reward_from_succ_prespective(succ_node,rollout_node)
        return delta_subtree_R

    def _backup_successor_node(self,succ_node):
        parent_node = succ_node.get_parent_node()
        assert parent_node is not None
        if succ_node.get_player() == parent_node.get_player():
                raise ValueError("Successor should have different player")
                delta_W = successor.delta_subtree_R +  successor.get_parent_reward()
                successor.W += delta_W
                node.delta_subtree_R += delta_W
        else:
            delta_W = succ_node.delta_subtree_R +  -1*succ_node.get_parent_reward()
            succ_node.W += delta_W
            parent_node.delta_subtree_R += -1*delta_W
            
        succ_node.N += 1
        succ_node.delta_subtree_R = 0



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

    def initialize_node_attributes(self,node):
        super().initialize_node_attributes(node)
        node.P = 0


    def expand(self,node):
        if not node.is_terminal():
            successor = self._initialize_all_successors_and_chooses_random_node(node)
            successor.delta_subtree_R = successor.delta_subtree_R + \
                                        self._simulate_from_random_node(successor)
            self._backup_successor_node(successor)
        else:
            raise ValueError("shouldn't be terminal")
            
    def _initialize_all_successors_and_chooses_random_node(self,node):
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
        ALPHAZERO ONE
        Uses node.N, node.W, node.P
    '''
    def __init__(self,network,device):
        super().__init__()
        self.network = network
        self.device = device

    def initialize_node_attributes(self,node):
        node.N = 0
        node.W = 0
        node.delta_subtree_R = 0
        node.P = 0

    def expand(self,node):
        if not node.is_terminal():
            self._initialize_policy_on_successors_and_estimate_node_value(node)
        else:
            raise ValueError("shouldn't be terminal")

    def _initialize_policy_on_successors_and_estimate_node_value(self,node):
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




class Normal_With_Network_Estimation(One_Successor_Rollout):
    '''
        Uses node.N, node.W
    '''
    def __init__(self,network,device):
        super().__init__()
        self.network = network
        self.device = device

    def expand(self,node):
        if not node.is_terminal():
            successor = self._choose_random_successor(node)
            successor.delta_subtree_R += self._estimate_successor(successor)
            self._backup_successor_node(successor)
        else:
            raise ValueError("shouldn't be terminal")

    def initialize_node_attributes(self,node):
        node.N = 0
        node.W = 0

    def _estimate_node(self,node):
        if node.is_terminal():
            raise ValueError("Should not be terminal") 
        current_board = node.get_current_observation()
        x = torch.from_numpy(current_board).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            self.network.load_state(x)
            estimate = self.network.get_state_value()
        return estimate

    def _backup_successor_node(self,succ_node):
        parent_node = succ_node.get_parent_node()
        assert parent_node is not None
        if succ_node.get_player() == parent_node.get_player():
                raise ValueError("Successor should have different player")
                delta_W = successor.delta_subtree_R +  successor.get_parent_reward()
                successor.W += delta_W
                node.delta_subtree_R += delta_W
        else:
            delta_W = succ_node.delta_subtree_R +  -1*succ_node.get_parent_reward()
            succ_node.W += delta_W
            parent_node.delta_subtree_R += -1*delta_W
            
        succ_node.N += 1
        succ_node.delta_subtree_R = 0
