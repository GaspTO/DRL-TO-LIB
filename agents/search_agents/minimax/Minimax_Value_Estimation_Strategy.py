from environments.core.Players import Players, Player, IN_GAME, TERMINAL, TIE_PLAYER_NUMBER
import numpy as np
import torch
import random


#! tip: why not have a hash table with values working as a cache?

""" Expand """
class Minimax_Value_Estimation_Strategy:
    def __init__(self):
        self.parent_node = None

    def load_parent_node(self,parent_node):
        return NotImplementedError

    def estimate_node(self,node):
        return NotImplementedError

'''
Handles total_visits through rollout estimate
Handles total_value
'''
class Random_Rollout_Estimation(Minimax_Value_Estimation_Strategy):
    '''
        NO POLICY BIAS.
        EVERY expand CALL CREATES ONLY ONE SUCCESSOR, WHOSE NODE.total_value IS ESTIMATED BY A RANDOM ROLLOUT
    '''
    def __init__(self,num_rollouts=1):
        super().__init__()
        self.num_rollouts = num_rollouts

    def estimate_node(self,node):
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



class Network_Value_Estimation(Minimax_Value_Estimation_Strategy):
    '''
        NO POLICY BIAS
        STATE ESTIMATE IS DONE USING NETWORK
    '''
    def __init__(self,network,device):
        super().__init__()
        self.network = network
        self.device = device

    def estimate_node(self,node):
        if node.is_terminal():
            return 0
        else:
            current_board = node.get_current_observation()
            with torch.no_grad():
                self.network.load_observations(np.array([current_board]))
                estimate = self.network.get_state_value()
            return estimate


class Network_Q_Estimation(Minimax_Value_Estimation_Strategy):
    '''
        NO POLICY BIAS
        STATE ESTIMATE IS DONE USING NETWORK BUT IT NEEDS TO BE LOADED WHEN PASSED IN __init__
    '''
    def __init__(self,network,device):
        super().__init__()
        self.network = network
        self.device = device
        self.parent_node = None
        self.q_estimations = None
       
    def estimate_node(self,node):
        if self.parent_node != node.get_parent_node():
            self.parent_node = node.get_parent_node()
            with torch.no_grad():
                current_board = self.parent_node.get_current_observation()
                self.network.load_observations(np.array([current_board]))
                self.q_estimations = self.network.get_q_values()[0]
        if node.is_terminal():
            return 0
        else:
           return self.q_estimations[node.get_parent_action()]