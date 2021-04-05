'''
import sys, os
sys.path.append(dirname(dirname(abspath(__file__))))
sys.path.append("/home/nizzel/Desktop/Tiago/Computer_Science/Tese/DRL-TO-LIB")
from agents.tree_agents.Search_Node import Search_Node


class Astar_Search():
    def __init__(self):



    
    def node_initializer(node):
        assert hasattr(node,'num_wins') == False
        node.num_wins = 0
        assert hasattr(node,'num_losses') == False
        node.num_losses = 0
        assert hasattr(node,'num_draws') == False
        node.num_draws = 0
        assert hasattr(node,'num_chosen_by_parent') == False
        node.num_chosen_by_parent = 0
        assert hasattr(node,'belongs_to_tree') == False
        node.belongs_to_tree = False


'''



