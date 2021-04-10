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
from agents.Agent import Agent
import numpy as np



class Tree_Search_Iteration(Agent):
    def __init__(self,environment,playout_iterations,search_expansion_iterations,debug=False):
        self.environment = environment
        self.playout_iterations = playout_iterations
        self.search_expansion_iterations = search_expansion_iterations
        self.debug = debug
        

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
                winner = self.current_node.get_winner()
            else:
                winner = self.expand_tactic.expand()
            self.backtrack(winner)
            if self.debug: print("summarize:" + str(self.root.W/self.root.N))

    def backtrack(self,winner):
        def update_statistic(current_node,winner):
            current_node.num_chosen_by_parent += 1
            if TIE_PLAYER_NUMBER == winner.get_number():
                current_node.num_draws += 1
            elif current_node.get_current_player().get_number() == winner.get_number():
                current_node.num_wins += 1
            else:
                current_node.num_losses += 1

        while not self.current_node.is_root():
            update_statistic(self.current_node,winner)
            self.current_node = self.current_node.get_parent_node()
        assert self.current_node.is_root()
        update_statistic(self.current_node,winner)
        

    """
    Agent Interface
    """
    def play(self,observation=None):
        if observation is None:
            observation = self.environment.get_current_observation()
        self.root =  Search_Node(self.environment,observation,initializer_fn=None)
        self.root.belongs_to_tree = True
        for _ in range(self.playout_iterations):
            self.playout(k=self.search_expansion_iterations)
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
        if node.num_chosen_by_parent == 0:
            return 0.  # avoid unseen moves
        return (node.num_losses + 0.5*node.num_draws) / node.num_chosen_by_parent

        
            
            






''' MAIN '''
env = Custom_K_Row(board_shape=3, target_length=3)
agent = Tree_Search_Iteration(env,playout_iterations=100,search_expansion_iterations=1)

#* tree policy 
agent.set_tree_policy_tactic(Greedy_DFS(agent))
#agent.set_tree_policy_tactic(Greedy_Best_First_Search(agent))
#agent.set_tree_policy_tactic(Adversarial_Greedy_Best_First_Search(agent))
#* expand policy
start = time.time()
agent.set_expansion_tactic(One_Successor_Rollout(agent))
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
