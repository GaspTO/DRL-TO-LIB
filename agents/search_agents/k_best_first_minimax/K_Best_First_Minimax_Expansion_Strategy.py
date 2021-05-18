import torch
import numpy as np




""" Expand """
class K_Best_First_Minimax_Expansion_Strategy:
    def __init__(self):
        pass

    ''' generates successors of all nodes.
    nodes is a list so that we can parallelize '''
    def expand(self,nodes:list) -> float:
        self._validate_input(nodes)
        self._instanciate_successors(nodes)

    def _validate_input(self,nodes:list):
        if len(nodes) == 0:
            raise ValueError("Expanding no nodes")
        for node in nodes:
            if node.is_terminal():
                raise ValueError("Trying to expand terminal node")

    def _instanciate_successors(self,node):
        raise NotImplementedError


''' For successors, no exploration bias '''
class K_Best_First_All_Successors_Rollout(K_Best_First_Minimax_Expansion_Strategy):
    '''
        NO POLICY BIAS.
        EVERY expand CALL CREATES ONLY ONE SUCCESSOR, WHOSE NODE.total_value IS ESTIMATED BY A RANDOM ROLLOUT
    '''
    def __init__(self,num_rollouts=1):
        super().__init__()
        self.num_rollouts = num_rollouts

    def _instanciate_successors(self,nodes:list):
        for node in nodes:
            node.expand_rest_successors()
            for n in node.get_successors():
                if n.is_terminal():
                    n.value = 0
                    n.non_terminal_value = None 
                else:
                    n.value = self._rollout(n)
                    n.non_terminal_value = n.value

    #* returns the value of subtree total reward from the prespective of succ_node
    def _rollout(self,node):
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



class K_Best_First_Network_Successor_V(K_Best_First_Minimax_Expansion_Strategy):
    '''
        NETWORK POLICY BIAS (NODE.exploration_bias)
        EACH EXPANSION GENERATES ALL SUCCESSORS. EVERY SUCCESSOR RECEIVE A POLICY BIAS.
        AND THE NODE EXPANDED RECEIVES A NODE.total_value ESTIMATION.
    '''
    def __init__(self,network,device,batch_size=1):
        super().__init__()
        self.network = network
        self.device = device
        self.batch_size = batch_size

    def _instanciate_successors(self,nodes:list):
        child_nodes_queue = []
        observations = []
        child_nodes_state_values = []
        #* put child nodes that are not terminal to queue to be evaluated by network
        for node in nodes:
            child_nodes = node.expand_rest_successors()
            for child in child_nodes:
                if child.is_terminal():
                    child.value = 0
                    child.non_terminal_value = None
                else:
                    child_nodes_queue.append(child)
                    observations.append(child.get_current_observation())

        #* run network and add values to a list
        observations = torch.tensor(observations)
        for x in torch.split(observations,self.batch_size):
            with torch.no_grad():
                state_values = self.network.load_observations(x.numpy()).get_state_value()
                child_nodes_state_values.extend(state_values)
    
        #* put those results in the appropriate child nodes
        assert len(child_nodes_queue) == len(child_nodes_state_values)
        for i in range(len(child_nodes_queue)):
            child_nodes_queue[i].value = child_nodes_state_values[i].item()
            child_nodes_queue[i].non_terminal_value = child_nodes_queue[i].value
    




"""
class Network_Successor_V(Expansion_Strategy):
    '''
        NETWORK POLICY BIAS (NODE.exploration_bias)
        EACH EXPANSION GENERATES ALL SUCCESSORS. EVERY SUCCESSOR RECEIVE A POLICY BIAS.
        AND THE NODE EXPANDED RECEIVES A NODE.total_value ESTIMATION.
    '''
    def __init__(self,network,device):
        super().__init__()
        self.network = network
        self.device = device

    def expand(self,node):
        estimate = super().expand(node)
        return estimate

    def _instanciate_successors(self,node):
        child_nodes = node.expand_rest_successors()
        with torch.no_grad():
            for node_child in child_nodes:
                self.network.load_observations(np.array([node_child.get_current_observation()]))
                state_value = self.network.get_state_value()
                if node_child.is_terminal():
                    node_child.value = 0
                else:
                    node_child.value = state_value

    def _estimate_node(self, node):        
        return None
"""  

class  K_Best_First_Network_Successor_Q(K_Best_First_Minimax_Expansion_Strategy):
    '''
        NETWORK POLICY BIAS (NODE.exploration_bias)
        EACH EXPANSION GENERATES ALL SUCCESSORS. EVERY SUCCESSOR RECEIVE A POLICY BIAS.
        AND THE NODE EXPANDED RECEIVES A NODE.total_value ESTIMATION.
    '''
    def __init__(self,network,device):
        super().__init__()
        self.network = network
        self.device = device

    def expand(self,node):
        current_board = node.get_current_observation()
        with torch.no_grad():
            self.network.load_observations(np.array([current_board]))
        estimate = super().expand(node)
        return estimate

    def _instanciate_successors(self,node):
        child_nodes = node.expand_rest_successors()
        with torch.no_grad():
            q_values = self.network.get_q_values()
            for node_child in child_nodes:
                if node_child.is_terminal():
                    node_child.value = 0
                else:
                    node_child.value = q_values[0][node_child.parent_action].item()

    def _estimate_node(self, node):        
        return None
    



