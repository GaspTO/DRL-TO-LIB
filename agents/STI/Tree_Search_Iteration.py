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



class Tree_Search_Iteration(Agent):
    def __init__(self,environment,tree_policy,tree_expansion, playout_iterations,search_expansion_iterations,debug=False):
        self.environment = environment
        self.tree_policy_tactic = tree_policy
        self.expand_tactic = tree_expansion
        self.playout_iterations = playout_iterations
        self.search_expansion_iterations = search_expansion_iterations
        self.debug = debug
        
    def reset(self):
        self.tree_policy_tactic.reset(self.root)
        self.expand_tactic.reset()

    def run_n_playouts(self,n=1):
        for i in range(n):
            self.reset()
            node = self.tree_policy_tactic.forward()
            self.expand_tactic.update_tree(node)
            assert self.root.num_chosen_by_parent == i + 1


    """
    Agent Interface
    """
    def play(self,observation=None):
        if observation is None: observation = self.environment.get_current_observation()
        self.root =  Search_Node(self.environment,observation,initializer_fn=None)
        self.root.belongs_to_tree = True
        self.run_n_playouts(n=self.playout_iterations)
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
env.step(2)
env.step(1)
env.step(8)
env.step(7)


#* tree policy 
tree_policy = Greedy_DFS()
#agent.set_tree_policy_tactic(Greedy_Best_First_Search(agent))
#agent.set_tree_policy_tactic(Adversarial_Greedy_Best_First_Search(agent))
#* expand policy
tree_expansion = One_Successor_Rollout()


agent = Tree_Search_Iteration(env,playout_iterations=100,tree_policy=tree_policy,tree_expansion=tree_expansion,search_expansion_iterations=1)
start = time.time()
a = agent.play()
print("time.time=" + str(start-time.time()))
print("play:")
print(a)

