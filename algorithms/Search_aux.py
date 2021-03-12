import numpy
import os
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))

import numpy
from numpy.core.numeric import normalize_axis_tuple

#from environments.gym_gomoku.envs import state
from environments.Gomoku import GomokuEnv
from collections import deque
from math import sqrt,log
import random




class Abstract_MCTS_Search():
    def __init__(self,state,num_of_players=2):
        self.num_of_players = num_of_players
        self.players = ['PLAYER' + str(n) for n in range(num_of_players)]
        self.current_player = self.players[0]
        self.root = self.make_node(state)
        self.current_node = self.root

    #todo add phases as objects
    #todo make a general search that adds state instances to a queue and runs them
    #todo each state is something to do in search like a phases
    #todo phases can have complete control over the search class and they can change queue

    def make_node(self,state):
        return Node(state)

    def set_node_predifined_contents(self,node,belongs_to_tree=True):
        return NotImplemented()

    def set_current_node(self,node):
        self.current_node = node

    def find_action_for_root(self,num_iter=1):
        self.run_n_playouts(num_iter)
        action_node = self.play_best_move(self.root)
        return action_node.get_parent_action()

    def run_n_playouts(self,num_iter=1):
        for _ in range(num_iter):
            leaf_node = self.selection_phase(self.root)
            new_child_node = self.expansion_phase(leaf_node)
            terminal_node = self.simulation_phase(new_child_node)
            self.backpropagate(terminal_node)

    def selection_phase(self,node):
        while node.is_completely_expanded():
            node = self.selection_exploit_explore_criteria(node)
            self.set_current_node(node)
        return node

    def selection_exploit_explore_criteria(self,node):
        return NotImplemented()
       
    def expansion_phase(self,node):
        if not node.is_terminal():
             node = self.expansion_explore_criteria(node)
             self.set_node_predifined_contents(node)
        self.set_current_node(node) #placebo might be needed
        return node
        
    def expansion_explore_criteria(self,node):
        return NotImplemented()
            
    def simulation_phase(self,node):
        while not node.is_terminal():
            node = self.simulation_fast_generation_policy(node)
            self.set_current_node(node)
        return node

    def simulation_fast_generation_policy(self,node):
        return NotImplemented()

    def backpropagation_phase(self,node):
        self.last_node = node
        self.content_to_be_backpropagated = self.backpropagation_get_content_to_be_backpropagated(self.last_node)
        while not node.is_root():
            self.backpropagation_update_nodes(node,self.content_to_be_backpropagated,self.last_node)
            node = node.get_parent_node()
            self.set_current_node(node)

    def backpropagation_get_content_to_be_backpropagated(self,node):
        return NotImplemented()

    def backpropagation_update_nodes(self,node,last_node):
        return NotImplemented()

    def backpropagation_update_nodes(self,node):
        return NotImplemented()


        


class Two_Player_Standard_MCTS_Search(Abstract_MCTS_Search):
    def __init__(self):
        Abstract_MCTS_Search.__init__(self,num_of_players = 2)
        self.exploration_weight = 1
        
    def set_node_predifined_contents(self,node):
        node.set_content_item["num_visits"] = 0
        node.set_content_item["num_wins"] = 0
        node.set_content_item["belongs_to_tree"] = 0
        node.set_content_item["player"] = self.current_player
        
    def selection_exploit_explore_criteria(self,node):
        log_N_vertex = log(node.get_content_item("num_visits"))
        def uct(node):
            return node.get_content_item("num_wins") / node.get_content_item("num_visits") + \
                self.exploration_weight * sqrt(log_N_vertex / node.get_content_item("num_visits"))
        return max(node.get_children(), key=uct)

    def expansion_explore_criteria(self,node):
        new_node = node.expand_one_child()
        new_node.set_content_item("player",self.current_player)

    def simulation_fast_generation_policy(self,node):
        node = node.generate_any_child()
        node.set_content_item["belongs_to_tree"] = False

    def backpropagation_get_content_to_be_backpropagated(self,node):
        return NotImplemented()

    def backpropagation_update_nodes(self,node,last_node):
        if node.get_content_item["player"] == last_node.get_content_item["player"]:
            node.set_content_item["num_wins"] += 1
        node.get_content_item["num_visits"] += 1

    
        
        

 

''' Search Engine '''
class MCTS_Search():
    def __init__(self,root,num_of_players=2, blocks = None, initial_variables = {}):
        self.root = root
        self.current_node = self.root
        self.variables = initial_variables
        self.num_of_players = num_of_players
        self.players = ['PLAYER' + str(n) for n in range(num_of_players)]
        if blocks is None: raise ValueError("Need To Define MCTS phase Blocks")
        self.Selection_Phase_Block = blocks['Selection_Phase_Block']
        self.Expansion_Phase_Block = blocks['Expansion_Phase_Block']
        self.Simulation_Phase_Block = blocks['Simulation_Phase_Block']
        self.Backpropagation_Phase_Block = blocks['Backpropagation_Phase_Block']

    def run_n_playouts(self,iterations):
        for _ in range(iterations):
            self.Selection_Phase_Block.run(self)
            self.Expansion_Phase_Block.run(self)
            self.Simulation_Phase_Block.run(self)
            self.Backpropagation_Phase_Block.run(self)



''' Abstract Block '''
class Block():
    def run(search):
        return NotImplemented() 

''' Abstract MCTS Blocks '''
class Selection_Phase_Block(Block):
    def __init__(self,blocks = None):
        if blocks is None: raise ValueError("Need To Define Selection phase Blocks")
        self.exploitation_exploration_criteria = blocks['exploitation_exploration_criteria']
        
    def run(self,search):
        while search.current_node.is_completely_expanded():
            search.current_node = self.exploitation_exploration_criteria.run(search)


class Expansion_Phase_Block(Block):
    def __init__(self,blocks = None):
        if blocks is None: raise ValueError("Need To Define Expansion phase Blocks")
        self.expansion_criteria = blocks['expansion_criteria']

    def run(self,search):
        if not search.current_node.is_terminal():
            search.current_node = self.expansion_criteria.run(search)


class Simulation_Phase_Block(Block):
    def __init__(self,blocks = None):
        if blocks is None: raise ValueError("Need To Define Simulation phase Blocks")
        self.fast_generation_policy = blocks['fast_generation_policy']

    def run(self,search):
        while not search.current_node.is_terminal():
            search.current_node = self.fast_generation_policy.run(search)


class Backpropagation_Phase_Block(Block):
    def __init__(self,blocks = None):
        if blocks is None: raise ValueError("Need To Define Backpropagation phase Blocks")
        self.backpropagate_update_nodes = blocks['backpropagate_update_nodes']

    def run(self,search):
        search.variables['last_node'] = search.current_node
        while not search.current_node.is_root():
            self.backpropagate_update_nodes.run(search)
            search.current_node = search.current_node.get_parent_node()

''' Concrete '''




print("MAIN")
search = Search(initial_blocks=[MCTS_Block()])
print("END MAIN")

""""""""""""""""""""""""""""""""""""""

''' Strategy Collections '''
class MCTS_Strategy_Collection():
    ''' Selection '''
    class Exploitation_exploration_criteria(Block):
        def run(self,search):
            return NotImplemented()
    ''' Expansion '''
    class Expansion_criteria(Block):
        def run(self,search):
            return NotImplemented()
    
    ''' Simulation '''
    class Fast_generation_policy(Block):
        def run(self,search):
            return NotImplemented()

    ''' Backpropagation '''
    class Get_content_to_be_backpropagated(Block):
        def run(self,search):
            return NotImplemented()

    class Backpropagate_update_nodes(Block):
        def run(self,search):
            return NotImplemented()
    

class MCTS_Gomoku_Factory():
    ''' Selection '''
    class Exploitation_exploration_criteria(Block):
        def run(self,search):
            log_N_vertex = log(search.current_node.get_content_item("num_visits"))
            def uct(node):
                return node.get_content_item("num_wins") / node.get_content_item("num_visits") + \
                    self.exploration_weight * sqrt(log_N_vertex / node.get_content_item("num_visits"))
            return max(search.current_node.get_children(), key=uct) 

    ''' Expansion '''
    class Expansion_criteria(Block):
        def run(self,search):
            return search.current_node.expand_one_child()

    ''' Simulation '''
    class Fast_generation_policy(Block):
        def run(self,search):
            node = search.current_node.generate_any_child()
            search.current_node.set_content_item["belongs_to_tree"] = False

    ''' Backpropagation '''
    class Backpropagate_update_nodes(Block):
        def run(self,search):
            if search.current_node.get_depth() % search.num_of_players \
            == search.variables["last_node"].get_depth() % search.mum_of_players():
                num_wins = search.current_node.get_content_item("num_wins")
                if num_wins is None: num_wins = 0
                num_visits = search.current_node.get_content_item("num_visits")
                if num_visits is None: num_visits = 0
                search.current_node.set_content_item("num_wins",num_wins + 1)
                search.current_node.set_content_item("num_visits",num_visits + 1)

    
    def get_search_engine(self):
        gomoku_blocks = {
            'exploitation_exploration_criteria': self.Exploitation_exploration_criteria(),
            'expansion_criteria': self.Expansion_criteria(),
            'fast_generation_policy': self.Fast_generation_policy(),
            'backpropagate_update_nodes': self.Backpropagate_update_nodes()
        }
        root = Node()
        engine = MCTS_Search(root,num_of_players=2, blocks = gomoku_blocks)



    


    


















""""""""""""""""""""""""""""""""""""""

class MCTS_Gomoku_Search():
    positive_color = 'black' #me?
    negative_color = 'white'
    
    def __init__(self,env,state):
        self.root = MCTS_Node(env,state)
        self.exploration_weight = 1

    ''' call this from the outside. It will run and play '''
    def find_action_for_root(self,n=1):
        self.run(n)
        action_node = MCTS_Gomoku_Search.play_node(self.root)
        print("(" + str(self.root.num_wins) + "," + str(self.root.num_visits) + ")" )
        return action_node.mother_action

    def run(self,n=1):
        for _ in range(n):
            leaf = MCTS_Gomoku_Search.selection_phase(self.root)
            terminal_node = MCTS_Gomoku_Search.simulation_phase(leaf)
            MCTS_Gomoku_Search.backpropagate(terminal_node)

    def selection_phase(node):
        while(node.is_expanded() and not node.is_terminal()):
            node = MCTS_Gomoku_Search.uct_select(node)
        if(not node.is_terminal()):
            node = node.expand_one_child()
        return node

    def play_node(node):
        if(node.is_terminal()):
            raise ValueError("Chose a leaf")
        else:
            def score(node):
                if node.num_visits == 0:
                    return float("-inf")  # avoid unseen moves
                return node.num_wins / node.num_visits  # average reward

            return max(node.get_children(), key=score)

    def uct_select(node,exploration_weight=1):
        if(not node.expanded): raise ValueError("node not expanded")

        log_N_vertex = log(node.num_visits)
        def uct(node):
            "Upper confidence bound for trees"
            return node.num_wins / node.num_visits + \
                exploration_weight * sqrt(log_N_vertex / node.num_visits) #todo hummm make sure this is the formula

        return max(node.get_children(), key=uct)        

    def simulation_phase(node):
        while(not node.is_terminal()):
            node = node.generate_any_child()
            node.belongs_to_tree = False
        return node
        
    def backpropagate(node): 
        #win = True if MCTS_Gomoku_Search.positive_color == node.state.color else False
        win = True if MCTS_Gomoku_Search.positive_color != node.state.color else False #todo make sure this is not gonna cause problem
        while(node != None):
            if(node.belongs_to_tree):
                if(node.state.color == MCTS_Gomoku_Search.positive_color and win): #my node and I won
                    node.num_wins += 1
                elif(node.state.color != MCTS_Gomoku_Search.positive_color and not win): #he's node and he won
                    node.num_wins += 1
                node.num_visits += 1
            node = node.parent




    












""" TEST """
test = False
test_basic_search = False
test_mcts = True
if(test):
    """ BASIC SEARCH """
    if(test_basic_search):
        env = GomokuEnv('black','random',9)
        init_state = env.state
        tree = Basic_Gomoku_Search(init_state)
        result = tree.BFS(2)
        result = tree.DFS()
        print(result)

    """ MCTS """
    if(test_mcts):
        env = GomokuEnv('black','beginner',9)
        mcts_search = MCTS_Gomoku_Search(env,env.state)
        play = mcts_search.find_action_for_root(10000)
        print(play)
        #mcts_search.run(1000)
        #print("kappo")
