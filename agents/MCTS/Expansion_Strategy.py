from environments.core.Players import Players, Player, IN_GAME, TERMINAL, TIE_PLAYER_NUMBER
import torch
import random





""" Expand """
class Expansion_Strategy:
    def __init__(self):
        pass

    ''' generates successors and returns value
    of node to propagate back'''
    def expand(self,node) -> float:
        if not node.is_terminal():
            self._instanciate_successors(node)
            estimate = self._estimate_node(node)
            return estimate
        else:
            raise ValueError("shouldn't be terminal")

    ''' creates successors and instanciates
     any relevant attributes '''
    def _instanciate_successors(self,node):
        return NotImplementedError

    ''' estimates value of node and returns it'''
    def _estimate_node(self,node) -> float:
        return NotImplementedError


'''
Handles total_visits through rollout estimate
Handles total_value
'''
class One_Successor_Rollout(Expansion_Strategy):
    '''
        NO POLICY BIAS.
        EVERY expand CALL CREATES ONLY ONE SUCCESSOR, WHOSE NODE.total_value IS ESTIMATED BY A RANDOM ROLLOUT
    '''
    def __init__(self,num_rollouts=1):
        super().__init__()
        self.num_rollouts = num_rollouts

    def _instanciate_successors(self,node):
        node.expand_rest_successors()
       
    #* returns the value of subtree total reward from the prespective of succ_node
    def _estimate_node(self,node):
        accumulated_reward = 0.
        for i in range(self.num_rollouts):
            rollout_node = node
            while not rollout_node.is_terminal():
                parent_player = rollout_node.get_player()
                rollout_node = rollout_node.find_random_successor()
                if node.get_player() == parent_player:
                    accumulated_reward += rollout_node.get_parent_reward()
                else:
                    accumulated_reward += -1*rollout_node.get_parent_reward()
        return accumulated_reward/self.num_rollouts


class Network_Value(Expansion_Strategy):
    '''
        NO POLICY BIAS
        STATE ESTIMATE IS DONE USING NETWORK
    '''
    def __init__(self,network,device):
        super().__init__()
        self.network = network
        self.device = device

    def _instanciate_successors(self,node):
        node.expand_rest_successors()

    def _estimate_node(self,node):
        if node.is_terminal():
            raise ValueError("shouldn't be terminal")
        else:
            current_board = node.get_current_observation()
            x = torch.from_numpy(current_board).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                self.network.load_state(x)
                estimate = self.network.get_state_value()
        return estimate


"""
THIS IS THE MCTS RL
"""
class Network_Policy_One_Successor_Rollout(One_Successor_Rollout):
    '''
        NETWORK POLICY BIAS (NODE.policy_value)
        AFTER EXPANDING, CHOOSES ONE RANDOM NODE AND DOES RANDOM ROLLOUT TO ESTIMATE ITS VALUE
    '''
    def __init__(self,network,device):
        super().__init__()
        self.network = network
        self.device = device

    def expand(self,node):
        current_board = node.get_current_observation()
        st = torch.from_numpy(current_board).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            self.network.load_state(st)
        estimate = super().expand(node)
        return estimate
        
    def _instanciate_successors(self,node):
        child_nodes = node.expand_rest_successors()
        policy_values = self.network.get_policy_values(apply_softmax=True, mask=torch.tensor(node.get_mask()))
        for node_child in child_nodes:
            node_child.policy_value = policy_values[0][node_child.parent_action].item()



class Network_Policy_Value(Expansion_Strategy):
    '''
        NETWORK POLICY BIAS (NODE.policy_value)
        EACH EXPANSION GENERATES ALL SUCCESSORS. EVERY SUCCESSOR RECEIVE A POLICY BIAS.
        AND THE NODE EXPANDED RECEIVES A NODE.total_value ESTIMATION.
    '''
    def __init__(self,network,device):
        super().__init__()
        self.network = network
        self.device = device

    def expand(self,node):
        current_board = node.get_current_observation()
        st = torch.from_numpy(current_board).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            self.network.load_state(st)
        estimate = super().expand(node)
        return estimate

    def _instanciate_successors(self,node):
        child_nodes = node.expand_rest_successors()
        policy_values = self.network.get_policy_values(apply_softmax=True, mask=torch.tensor(node.get_mask()))
        for node_child in child_nodes:
            node_child.policy_value = policy_values[0][node_child.parent_action].item()

    def _estimate_node(self, node):
        with torch.no_grad():
            estimate = self.network.get_state_value()
        return estimate
    
    



