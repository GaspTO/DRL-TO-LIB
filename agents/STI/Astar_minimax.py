from math import sqrt, log
import heapq
from agents.STI.Search_Evaluation_Function import UCT, PUCT
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
sys.path.append("/home/nizzel/Desktop/Tiago/Computer_Science/Tese/DRL-TO-LIB")


from abc import abstractmethod
from random import randint
from typing import AbstractSet
from math import sqrt,log
from agents.STI.Search_Node import Search_Node
from agents.STI.Tree_Policy import Tree_Policy, Greedy_DFS, Adversarial_Greedy_Best_First_Search
from agents.STI.Expansion_Strategy import Expansion_Strategy, One_Successor_Rollout


from environments.Custom_K_Row import Custom_K_Row
from environments.core.Players import Players, Player, IN_GAME, TERMINAL, TIE_PLAYER_NUMBER
import time
from agents.Agent import Agent
import numpy as np
import torch
import random



class Astar_minimax(Agent):
    def __init__(self,environment,network,device,debug=False):
        self.environment = environment
        self.debug = debug
        self.network = network
        self.device = device

    def play(self,observation=None):
        if observation is None: observation = self.environment.get_current_observation()
        self.root =  Search_Node(self.environment,observation,initializer_fn=None)
        self.root.belongs_to_tree = True
        self.reset()
        return self.forward()
    
    def forward(self):
        while True:
            if self.node.is_terminal():
                return self.find_root_play(self.node)

            if not self.node.is_completely_expanded() and not self.node.is_terminal():
                self.expand(self.node)

            #* 1. I'm ally, put in adversary successors in greedy frontier, 
            if self.node.get_current_player() == self.root.get_current_player():
                for n in self.node.get_successors(): 
                    self.push_greedy_frontier(n)  
                self.node =  heapq.heappop(self.greedy_frontier)[-1]
                assert self.node.get_current_player() != self.root.get_current_player()

            #* 2. I'm adversary, choose best local ally
            elif self.node.get_current_player() != self.root.get_current_player():
                self.node = max(self.node.get_successors(),key=lambda node: node.p)
                assert self.node.get_current_player() == self.root.get_current_player()

    def push_greedy_frontier(self,node):
        heapq.heappush(self.greedy_frontier,(-1*node.p,self.i,node)) #-1 cause piority is decreasing
        self.i += 1

    def reset(self):
        self.node = self.root
        self.i = 0
        self.greedy_frontier = []

    def expand(self,node):
        ''' expand all and adds node.p '''
        nodes = node.expand_rest_successors()
        current_board = node.get_current_observation()
        x = torch.from_numpy(current_board).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            p = self.network(x,torch.tensor(node.get_mask()),False)
            p = torch.softmax(p,dim=1)
        for node_child in nodes:
            if node_child.is_terminal():
                if node_child.get_winner() == node.get_current_player():
                    node_child.p = 1
                elif node_child.get_winner() == node_child.get_current_player():
                    node_child.p = -1
                else:
                    node_child.p = 0
            else:
                node_child.p = p[0][node_child.parent_action]
            node_child.belongs_to_tree = True

    def find_root_play(self,node):
        while not node.get_parent_node().is_root():
            node = node.get_parent_node()
        return node.get_parent_action()
