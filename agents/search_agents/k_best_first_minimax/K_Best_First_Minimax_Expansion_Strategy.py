from agents.Neural_Agent import Neural_Agent
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


class Network_K_Best_First_Minimax_Expansion_Strategy(K_Best_First_Minimax_Expansion_Strategy):
    def __init__(self,network,batch_size):
        super().__init__()
        self.network = network
        self.device = self.network.device
        self.batch_size = batch_size

    def set_network(self,network):
        self.network = network
        self.device = self.network.device
    
    def set_batch_size(self,batch_size):
        self.batch_size = batch_size


''' For successors, no exploration bias '''
class K_Best_First_All_Successors_Rollout_Expansion_Strategy(K_Best_First_Minimax_Expansion_Strategy):
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



class K_Best_First_Network_Successor_V_Expansion_Strategy(Network_K_Best_First_Minimax_Expansion_Strategy):
    '''
        NETWORK POLICY BIAS (NODE.exploration_bias)
        EACH EXPANSION GENERATES ALL SUCCESSORS. EVERY SUCCESSOR RECEIVE A POLICY BIAS.
        AND THE NODE EXPANDED RECEIVES A NODE.total_value ESTIMATION.
    '''
    def __init__(self,network,batch_size):
        super().__init__(network,batch_size)

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
        if len(child_nodes_queue) > 0:
            observations = torch.tensor(observations,device=self.device)
            
            if self.batch_size is not None:
                observations = torch.split(observations,self.batch_size)
            else:
                observations = observations.unsqueeze(0)

            for x in observations:
                with torch.no_grad():
                    state_values = self.network(x)
                    child_nodes_state_values.extend(state_values)
        
            #* put those results in the appropriate child nodes
            assert len(child_nodes_queue) == len(child_nodes_state_values)
            for i in range(len(child_nodes_queue)):
                child_nodes_queue[i].value = child_nodes_state_values[i].item()
                child_nodes_queue[i].non_terminal_value = child_nodes_queue[i].value
    


class K_Best_First_Network_Successor_Q_Expansion_Strategy(Network_K_Best_First_Minimax_Expansion_Strategy):
    '''
        NETWORK POLICY BIAS (NODE.exploration_bias)
        EACH EXPANSION GENERATES ALL SUCCESSORS. EVERY SUCCESSOR RECEIVE A POLICY BIAS.
        AND THE NODE EXPANDED RECEIVES A NODE.total_value ESTIMATION.
    '''
    def __init__(self,network,batch_size):
        super().__init__(network,batch_size)

    def _instanciate_successors(self,nodes:list):
        parent_nodes_queue = []
        observations = []
        parent_nodes_q_values = []
        #* put child nodes that are not terminal to queue to be evaluated by network


        for node in nodes:
            node.expand_rest_successors()
            parent_nodes_queue.append(node)
            observations.append(node.get_current_observation())
            
        #* run network and add values to a list
        if len(observations) > 0:
            observations = torch.tensor(observations,device=self.device)
            if self.batch_size is not None:
                observations = torch.split(observations,self.batch_size)
            else:
                observations = observations.unsqueeze(0)

            for x in observations:
                with torch.no_grad():
                    q_values = self.neural_agent.load_observations(x.numpy()).get_q_values()
                    parent_nodes_q_values.extend(q_values)
        
            #* put those results in the appropriate child nodes
            for parent_idx in range(len(parent_nodes_queue)):
                successors = parent_nodes_queue[parent_idx].get_successors()
                parent_q_values = parent_nodes_q_values[parent_idx]
                for succ in successors:
                    if succ.is_terminal():
                        succ.value = 0
                        succ.non_terminal_value = None
                    else:
                        succ.value = parent_q_values[succ.get_parent_action()].item()
                        succ.non_terminal_value = succ.value






