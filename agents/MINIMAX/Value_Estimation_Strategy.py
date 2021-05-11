from environments.core.Players import Players, Player, IN_GAME, TERMINAL, TIE_PLAYER_NUMBER
import torch
import random


""" Expand """
class Value_Estimation_Strategy:
    def __init__(self):
        pass

    ''' estimates value of node and returns value'''
    def estimate(self,node) -> float:
        return NotImplementedError


'''
Handles total_visits through rollout estimate
Handles total_value
'''
class Random_Rollout_Estimation(Value_Estimation_Strategy):
    '''
        NO POLICY BIAS.
        EVERY expand CALL CREATES ONLY ONE SUCCESSOR, WHOSE NODE.total_value IS ESTIMATED BY A RANDOM ROLLOUT
    '''
    def __init__(self,num_rollouts=1):
        super().__init__()
        self.num_rollouts = num_rollouts

    #* returns the value of subtree total reward from the prespective of succ_node
    def estimate(self,node):
        if node.is_terminal():
            return 0
        else:
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


class Network_Value_Estimation(Value_Estimation_Strategy):
    '''
        NO POLICY BIAS
        STATE ESTIMATE IS DONE USING NETWORK
    '''
    def __init__(self,network,device):
        super().__init__()
        self.network = network
        self.device = device

    def estimate(self,node):
        if node.is_terminal():
            return 0
        else:
            current_board = node.get_current_observation()
            x = torch.from_numpy(current_board).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                self.network.load_state(x)
                estimate = self.network.get_state_value()
            return estimate



