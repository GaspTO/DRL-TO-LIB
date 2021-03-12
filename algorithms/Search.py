import numpy
import os
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
import numpy
from numpy.core.numeric import NaN, normalize_axis_tuple
from environments.Gomoku import GomokuEnv
from environments.K_Row import K_RowEnv
from algorithms.Node import Gomoku_MCTSNode, K_Row_MCTSNode, MCTS_FIRST_PLAYER, MCTS_TIE, MCTS_SECOND_PLAYER
from collections import deque
from math import sqrt,log
import random



      
''' Search Engine '''
class MCTS_Search():
    def __init__(self,root, initial_variables = {}):
        self.root = root
        self.root.set_content_item('times_parent_won', 0.)
        self.root.set_content_item('times_chosen_by_parent', 0.)
        self.root.set_content_item('belongs_to_tree',True)
        self.current_node = self.root
        self.variables = initial_variables
        self.exploration_weight = 1


    def run_n_playouts(self,iterations):
        for n in range(iterations):
            self.selection_phase()
            self.expansion_phase()
            self.simulation_phase()
            self.backpropagation_phase()
            if(n>= 300000):
                print(self.current_node.get_content_item('num_'))


    ''' Selection Phase '''
    def selection_phase(self):
        while self.current_node.is_completely_expanded() and not self.current_node.is_terminal():
            self.current_node = self.selection_criteria()
            #print("Sel:\n" + str(self.current_node.get_content_item('state').board) )

    def selection_criteria(self):
            log_N_vertex = log(self.current_node.get_content_item("times_chosen_by_parent"))
            def uct(node):
                return node.get_content_item("times_parent_won") / node.get_content_item("times_chosen_by_parent") + \
                    self.exploration_weight * sqrt(log_N_vertex / node.get_content_item("times_chosen_by_parent"))
            return max(self.current_node.get_successors(), key=uct) 

    ''' Expansion Phase '''
    def expansion_phase(self):
        if not self.current_node.is_terminal():
            self.current_node = self.expansion_criteria()
            #print("Ex:\n" + str(self.current_node.get_content_item('state').board) )

    def expansion_criteria(self):
            return self.current_node.expand_random_successor({'times_chosen_by_parent':0., 'times_parent_won':0., 'belongs_to_tree':True})

    ''' Simulation Phase '''
    def simulation_phase(self):
        while not self.current_node.is_terminal():
            self.current_node = self.fast_generation_policy()
            #print("Sim:\n" + str(self.current_node.get_content_item('state').board) )

    def fast_generation_policy(self):
            return self.current_node.find_random_unexpanded_successor({'belongs_to_tree':False})

    ''' Backpropagation Phase '''
    def backpropagation_phase(self):
        self.last_node = self.current_node
        winner = self.current_node.who_won()
        if(winner == MCTS_FIRST_PLAYER):
            self.winner = 0
        elif(winner == MCTS_SECOND_PLAYER):
            self.winner = 1
        elif(winner == MCTS_TIE):
            self.winner = NaN #no winner
        else:
            raise ValueError("Some weird value returned in who won in backpropagation phase")
        
        
        if(self.winner == -1):
            print("TIE")
            #print("TIE\n" + str(self.last_node.get_content_item('state').board))
        if(self.winner == 0):
            print("FIRST")
            #print("FIRST\n" + str(self.last_node.get_content_item('state').board))
        if(self.winner == 1):
            print("SECOND")
            #print("SECOND\n" + str(self.last_node.get_content_item('state').board))


        while not self.current_node.is_root():
            self.backpropagate_update_nodes()
            self.current_node = self.current_node.get_parent_node()
        self.backpropagate_update_nodes()

    def backpropagate_update_nodes(self):
        if self.current_node.get_content_item("belongs_to_tree") is False:
            return
        if self.winner == NaN:
            times_parent_won = self.current_node.get_content_item("times_parent_won")
            self.current_node.set_content_item("times_parent_won",times_parent_won + 0.5)
        elif self.winner == self.current_node.get_depth() % 2 and self.winner >= 0:
            times_parent_won = self.current_node.get_content_item("times_parent_won")
            self.current_node.set_content_item("times_parent_won",times_parent_won + 1)

        times_chosen_by_parent = self.current_node.get_content_item("times_chosen_by_parent")
        self.current_node.set_content_item("times_chosen_by_parent",times_chosen_by_parent + 1)




''' Gomoku 
env = GomokuEnv('black','random',9)
gom_node = Gomoku_Node(env.state)
search = MCTS_Search(gom_node,num_of_players=2)
search.run_n_playouts(5000)
print(search.root.get_content_item('num_wins'))
print(search.root.get_content_item('num_visits'))
'''


env = K_RowEnv(board_shape=3, target_length=3)
k_row_node = K_Row_MCTSNode(env.state)
search = MCTS_Search(k_row_node)
search.run_n_playouts(1000000)
print(search.root.get_content_item('num_wins'))
print(search.root.get_content_item('num_visits'))
print("hey")