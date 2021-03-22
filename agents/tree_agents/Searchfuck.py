import numpy
import os
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
import numpy
from numpy.core.numeric import NaN, normalize_axis_tuple
from agents.tree_agents.Node import Gomoku_MCTSNode, K_Row_MCTSNode, MCTS_FIRST_PLAYER, MCTS_TIE, MCTS_SECOND_PLAYER, MCTS_WIN, MCTS_LOSS
from collections import deque
from math import sqrt,log
import random
import torch


      
''' Search Engine '''
class MCTS_Search():
    def __init__(self,root, initial_variables = {}, debug = False):
        self.root = root
        self.root.belongs_to_tree = True
        self.current_node = self.root
        self.variables = initial_variables
        self.exploration_weight = 1
        self.debug = debug

    def play_action(self, n_iterations = 0, debug = False):
            self.run_n_playouts(n_iterations)
            def score(node):
                if node.num_chosen_by_parent == 0:
                    return float("-inf")  # avoid unseen moves
                return (node.num_losses + 0.5*node.num_draws) / node.num_chosen_by_parent  # choose the board that made the opponent lose the most
            best_action =  max(self.root.get_successors(), key=score).parent_action

            if(self.debug == True or debug == True):
                for n in self.root.get_successors():
                    print("action:" + str(n.parent_action) + " score:" + str(score(n)))
                print("best action chosen:" +  str(best_action))
            return best_action

    def run_n_playouts(self,iterations):
        for n in range(iterations):
            self.selection_phase()
            self.expansion_phase()
            self.simulation_phase()
            self.backpropagation_phase()


    ''' Selection Phase '''
    def selection_phase(self):
        while self.current_node.is_completely_expanded() and not self.current_node.is_terminal():
            self.current_node = self.selection_criteria()
            #print("Sel:\n" + str(self.current_node.get_content_item('state').board) )

    def selection_criteria(self):
            log_N_vertex = log(self.current_node.num_chosen_by_parent)
            def uct(node):
                assert node.num_chosen_by_parent == node.num_losses + node.num_draws + node.num_wins
                opponent_losses = node.num_losses + 0.5 * node.num_draws
                return opponent_losses / node.num_chosen_by_parent + \
                    self.exploration_weight * sqrt(log_N_vertex / node.num_chosen_by_parent)
            return max(self.current_node.get_successors(), key=uct) 

    ''' Expansion Phase '''
    def expansion_phase(self):
        if not self.current_node.is_terminal():
            self.current_node = self.expansion_criteria()
            #print("Ex:\n" + str(self.current_node.get_content_item('state').board) )

    def expansion_criteria(self):
            node = self.current_node.expand_random_successor()
            node.belongs_to_tree = True
            return node

    ''' Simulation Phase '''
    def simulation_phase(self):
        while not self.current_node.is_terminal():
            self.current_node = self.fast_generation_policy()
            #print("Sim:\n" + str(self.current_node.get_content_item('state').board) )

    def fast_generation_policy(self):
            node = self.current_node.find_random_unexpanded_successor()
            node.belongs_to_tree = False
            return node

    ''' Backpropagation Phase '''
    def backpropagation_phase(self):
        self.last_node = self.current_node
        winner = self.current_node.who_won()
        if(winner == MCTS_TIE):
            self.winner = None
        elif(winner == MCTS_WIN):
            self.winner = self.current_node.get_depth() % 2
            raise ValueError("Weird Value")
        elif(winner == MCTS_LOSS):
            self.winner = (self.current_node.get_depth() -1) % 2 
            assert self.winner >= 0
        '''
        if(winner == MCTS_FIRST_PLAYER):
            self.winner = 0
        elif(winner == MCTS_SECOND_PLAYER):
            self.winner = 1
        elif(winner == MCTS_TIE):
            self.winner = None #no winner
        else:
            raise ValueError("Some weird value returned in who won in backpropagation phase")
        '''
        
        if(self.debug == True):
            if(self.winner == None):
                print("TIE")
            if(self.winner == 0):
                print("FIRST")
            if(self.winner == 1):
                print("SECOND")



        while not self.current_node.is_root():
            self.backpropagate_update_nodes()
            self.current_node = self.current_node.get_parent_node()
        self.backpropagate_update_nodes()

    def backpropagate_update_nodes(self):
        if self.current_node.belongs_to_tree == False:
            return
        if self.winner == None:
            self.current_node.num_draws += 1
        elif self.winner == self.current_node.get_depth() % 2 and self.winner >= 0:
            self.current_node.num_wins += 1
        elif self.winner == (self.current_node.get_depth() + 1) % 2 and self.winner >= 0:
            self.current_node.num_losses += 1
        else:
            raise ValueError("Weird Value in backpropagate update node")
        self.current_node.num_chosen_by_parent += 1





'''
env = K_RowEnv(board_shape=3, target_length=3)
env.step(0)
k_row_node = K_Row_MCTSNode(env.state)
search = MCTS_Search(k_row_node,debug = True)
search.run_n_playouts(100000)
p = search.play_action(debug=True)
print("hey " + str(p))
print("c")
'''



''' Gomoku - Not going to work
env = GomokuEnv('black','random',9)
gom_node = Gomoku_Node(env.state)
search = MCTS_Search(gom_node,num_of_players=2)
search.run_n_playouts(5000)
print(search.root.get_content_item('num_wins'))
print(search.root.get_content_item('num_visits'))
'''


class MCTS_Search_attempt_muzero(MCTS_Search):
    def __init__(self,root, device, initial_variables = {}, debug = False):
        MCTS_Search.__init__(self,root,initial_variables,debug)
        self.device = device


    def get_probs(self):
        def score(node):
            if node.num_chosen_by_parent == 0:
                return 0. # avoid unseen moves
            return (node.num_losses + 0.5*node.num_draws) / node.num_chosen_by_parent
        succ = self.root.get_successors()
        succ.sort(key=lambda x: x.parent_action)
        probs = [score(node) for node in succ]
        return torch.tensor(probs)

    def play_action(self, n_iterations = 0,debug = False):
        def score(node):
            if node.num_chosen_by_parent == 0:
                return float("-inf")  # avoid unseen moves
            return (node.num_losses + 0.5*node.num_draws) / node.num_chosen_by_parent  # choose the board that made the opponent lose the most
        best_action =  max(self.root.get_successors(), key=score).parent_action
        return best_action

    def run_n_playouts(self,net,iterations):
        self.net = net
        super().run_n_playouts(iterations)


    def selection_criteria(self):
        if(self.net is None): raise ValueError("We need to define a plausible network")
        sqrt_N = sqrt(self.current_node.num_chosen_by_parent)
        def puct(node):
            assert node.num_chosen_by_parent == node.num_losses + node.num_draws + node.num_wins
            opponent_losses = node.num_losses + 0.5 * node.num_draws
            U = self.exploration_weight * node.p * sqrt_N /(1 + node.num_chosen_by_parent)
            Q = opponent_losses/(node.num_chosen_by_parent + 1)
            return U + Q
        max_node =  max(self.current_node.get_successors(), key=puct)
        print("selection: " + str(max_node.parent_action))
        return max_node


    def expansion_criteria(self):
            nodes = self.current_node.expand_rest_successors()
            #current_board = self.current_node.state.board
            current_board = self.current_node.state.get_two_boards()
            x = torch.from_numpy(current_board).float().unsqueeze(0).to(self.device)
            p = self.net(x)
            for node in nodes:
                node.p = p[0][node.parent_action]
                node.belongs_to_tree = True
            return node