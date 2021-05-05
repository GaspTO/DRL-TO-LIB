import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
sys.path.append("/home/nizzel/Desktop/Tiago/Computer_Science/Tese/DRL-TO-LIB")


from abc import abstractmethod
from random import randint
from typing import AbstractSet
from math import sqrt,log
from environments.Custom_K_Row import Custom_K_Row
from environments.core.Players import Players, Player, IN_GAME, TERMINAL, TIE_PLAYER_NUMBER
import time
from agents.Agent import Agent
import numpy as np
import heapq
import torch

from agents.Simple_Astar.Search_Node import Search_Node





class ASTAR(Agent):
    def __init__(self,environment,network,debug=False):
        self.environment = environment
        self.network = network
        self.debug = debug
            
    def playout(self,k=1):
        for _ in range(k):
            self.forward()
            if not self.current_node.is_terminal():
                winner = self.expand_current()
            if len(self.greedy_frontier) == 0:
                break
        ''' debug:
        for n in self.root.get_successors():
            print(n.mega_choice)
        '''

    def reset(self):
        self.current_node = self.root
        self.i = 0
        self.greedy_frontier = []
        self.expand_current()
        assert self.current_node == self.root
        for n in self.current_node.get_successors():
            n.mega_choice = 0.
            n.mega_parent = n

    """
    ASTERISK SEARCH PART
    """
    def forward(self):
        while True:
            if not self.current_node.is_completely_expanded() and not self.current_node.is_terminal():
                if self.debug: print("\n")
                return

            #* 1. I'm ally, put in adversary successors in greedy frontier, 
            #* or it's terminal
            if self.current_node.get_current_player() == self.root.get_current_player() or self.current_node.is_terminal():
                for n in self.current_node.get_successors(): 
                    self.push_greedy_frontier(n)
                if len(self.greedy_frontier) == 0: return
                self.current_node =  heapq.heappop(self.greedy_frontier)[-1]
                self.current_node.mega_parent.mega_choice += 1
                if self.debug: print(str(self.current_node.get_parent_action()),end=" ")
                assert self.current_node.get_current_player() != self.root.get_current_player()

            #* 2. I'm adversary, choose best local ally
            elif self.current_node.get_current_player() != self.root.get_current_player():
                self.current_node = max(self.current_node.get_successors(),key=self.evaluate)
                if self.debug: print(str(self.current_node.get_parent_action()),end=" ")
                assert self.current_node.get_current_player() == self.root.get_current_player()
        if self.search.debug: print("")                

    def push_greedy_frontier(self,node):
        heapq.heappush(self.greedy_frontier,(-1*self.evaluate(node),self.i,node)) #-1 cause piority is decreasing
        self.i += 1

    def evaluate(self,node):
        return node.p
        '''
        log_N_vertex = log(node.get_parent_node().num_chosen_by_parent)
        assert node.num_chosen_by_parent == node.num_losses + node.num_draws + node.num_wins
        #opponent_losses = node.num_losses + 0.5 * node.num_draws
        opponent_losses = node.loss_value
        return opponent_losses / node.num_chosen_by_parent + self.exploration_weight * sqrt(log_N_vertex / node.num_chosen_by_parent)
        '''
    """
    EXPANDING PART
    """
    def expand_current(self):
        if not self.current_node.is_terminal():
            nodes = self.current_node.expand_rest_successors()
            current_board = self.current_node.get_current_observation()
            x = torch.from_numpy(current_board).float().unsqueeze(0).to("cpu")
            with torch.no_grad():
                p = self.network(x,torch.tensor(self.current_node.get_mask()),False)
                p = torch.softmax(p,dim=1)
            for node in nodes:
                node.p = p[0][node.parent_action]
                node.belongs_to_tree = True
                if not self.current_node.is_root():
                    node.mega_parent = self.current_node.mega_parent

    """
    Agent Interface
    """
    def play(self,observation=None):
        if observation is None:
            observation = self.environment.get_current_observation()
        self.root =  Search_Node(self.environment,observation,initializer_fn=None)
        self.root.belongs_to_tree = True
        self.current_node = self.root
        self.reset()
        self.playout(k=100)
        return self._get_best_action()

    def _get_best_action(self):
        return self._get_action_probabilities().argmax()

    def _get_action_probabilities(self):
        #the length of successors is not always the action_size 'cause invalid actions don't become successors
        action_probs = np.zeros(self.environment.get_action_size()) 
        for n in self.root.get_successors():
            action_probs[n.parent_action] = self._score_tactic(n)
        # if a vector is full of zeros 
        if(action_probs.sum() == 0.):
            for n in self.root.get_successors():
                action_probs[n.parent_action] = 1/len(self.root.get_successors()) 
        else:
            action_probs = action_probs/action_probs.sum()
        return action_probs

    def _score_tactic(self,node):
        return node.mega_choice

        
            
            






''' MAIN '''
"""
env = Custom_K_Row(board_shape=3, target_length=3)
agent = ASTAR(env)


#* expand policy
start = time.time()
a = agent.play()
print("time.time=" + str(start-time.time()))
print("play:")
print(a)
'''
start = time.time()
for i in range(10000000000):
    agent.playout(k=100)
    print("time.time=" + str(start-time.time()))
print("END")
''' 
"""