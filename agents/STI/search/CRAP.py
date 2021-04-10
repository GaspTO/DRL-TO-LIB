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
from agents.STI.Expansion_Stategy import Expansion_Strategy, One_Successor_Rollout
from environments.Custom_K_Row import Custom_K_Row
from environments.core.Players import Players, Player, IN_GAME, TERMINAL, TIE_PLAYER_NUMBER
import time















        


        



class Tree_Search_Iteration:
    def __init__(self,environment):
        self.environment = environment
        self.observation = self.environment.get_current_observation()
        self.root =  Search_Node(self.environment,self.observation,initializer_fn=None)
        self.root.belongs_to_tree = True

    def set_tree_policy_tactic(self,tree_policy_tactic):
        self.tree_policy_tactic = tree_policy_tactic

    def set_expansion_tactic(self,expansion_tactic):
        self.expand_tactic = expansion_tactic

    def reset(self):
        self.current_node = self.root
        self.tree_policy_tactic.reset()
        self.expand_tactic.reset()

    def playout(self,k=1):
        self.reset()
        for i in range(k):
            self.tree_policy_tactic.forward()
            if self.current_node.is_terminal():
                delta = -1 * self.current_node.get_parent_reward()
            else:
                delta = self.expand_tactic.expand()
            self.backtrack(delta)
            print("summarize:" + str(self.root.W/self.root.N))

    def backtrack(self,delta):
        p = self.current_node.get_current_player()
        self.current_node.W += delta
        self.current_node.N += 1
        while not self.current_node.is_root():
            self.current_node = self.current_node.get_parent_node()
            self.current_node.N += 1
            if self.current_node.get_current_player() == p:
                self.current_node.W += delta
            else:
                self.current_node.W -= delta
            assert self.current_node.parent_reward == 0

            
            
        
'''
    vl = TREEPOLICY(v0)
    if vl.terminal():
        vl.get_real_value()
    else:
        vl.expand()
    vl.backtrack()
'''


    


''' MAIN '''
env = Custom_K_Row(board_shape=3, target_length=3)
agent = Tree_Search_Iteration(env)
#* tree policy 
#agent.set_tree_policy_tactic(Greedy_DFS(agent))
#agent.set_tree_policy_tactic(Greedy_Best_First_Search(agent))
agent.set_tree_policy_tactic(Adversarial_Greedy_Best_First_Search(agent))
#* expand policy
agent.set_expansion_tactic(One_Successor_Rollout(agent))
start = time.time()
for i in range(10000000000):
    agent.playout(k=100)
    print("time.time=" + str(start-time.time()))
print("END")

    

