import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
from agents.Agent import Agent
from agents.tree_agents.MCTS_Search import MCTS_Search, MCTS_Node
import torch
from math import sqrt
import numpy as np
import random

class MCTS_RL_Search(MCTS_Search):
    def __init__(self,environment,network,device,observation = None,n_iterations=None,exploration_weight = 1.0,debug = False):
        super().__init__(environment,observation = observation,n_iterations = n_iterations,exploration_weight=exploration_weight,debug = debug)
        assert network is not None
        self.network = network
        self.device = device

    def get_play_probabilities(self, n_iterations = 0, debug = False):
        if self.n_iterations != None:
            assert self.n_iterations != None
            n_iterations = self.n_iterations
        self.run_n_playouts(n_iterations)
        sqrt_N = sqrt(self.current_node.num_chosen_by_parent)
        def puct(node):
            assert node.num_chosen_by_parent == node.num_losses + node.num_draws + node.num_wins
            opponent_losses = node.num_losses + 0.5 * node.num_draws
            U = self.exploration_weight * node.p * sqrt_N /(1 + node.num_chosen_by_parent)
            Q = opponent_losses/(node.num_chosen_by_parent + 1)
            return U + Q
        action_probs = np.zeros(self.environment.get_action_size()) #the len(successors) is not always the action_size
        for n in self.root.get_successors():
            action_probs[n.parent_action] = puct(n)
        action_probs = action_probs/action_probs.sum()
        if(np.isnan(action_probs).any()):
            print("crap")

        return action_probs

    def selection_criteria(self):
        if(self.network is None): raise ValueError("We need to define a plausible network")
        sqrt_N = sqrt(self.current_node.num_chosen_by_parent)
        def puct(node):
            assert node.num_chosen_by_parent == node.num_losses + node.num_draws + node.num_wins
            opponent_losses = node.num_losses + 0.5 * node.num_draws
            U = self.exploration_weight * node.p * sqrt_N /(1 + node.num_chosen_by_parent)
            Q = opponent_losses/(node.num_chosen_by_parent + 1)
            return U + Q
        max_node =  max(self.current_node.get_successors(), key=puct)
        return max_node

    def expansion_criteria(self):
            nodes = self.current_node.expand_rest_successors()
            current_board = self.current_node.get_current_observation()
            x = torch.from_numpy(current_board).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                p = self.network(x)
                p = torch.softmax(p,dim=1)
            for node in nodes:
                node.p = p[0][node.parent_action]
                node.belongs_to_tree = True
            random_idx = random.randint(0,len(nodes)-1)
            return nodes[random_idx]


class MCTS_RL_Agent(Agent):
    def __init__(self,environment,n_iterations,network,device,exploration_weight = 1.0,debug=False):
        super().__init__(environment)
        self.n_iterations = n_iterations
        self.network = network
        self.device = device
        self.exploration_weight = exploration_weight
        self.debug = debug

    def play(self,observation=None):
        if observation is None: observation = self.environment.get_current_observation()
        search = MCTS_RL_Search(self.environment,self.network,self.device,observation = observation,n_iterations=self.n_iterations,exploration_weight=self.exploration_weight,debug=self.debug)
        action = search.play_action(debug=False)
        return action

    def get_play_probabilities(self,observation=None):
        if observation is None: observation = self.environment.get_current_observation()
        search = MCTS_RL_Search(self.environment,self.network,self.device,observation = observation,n_iterations=self.n_iterations,exploration_weight=self.exploration_weight,debug=self.debug)
        probs = search.get_play_probabilities()
        return probs