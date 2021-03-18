import numpy
import os
import sys
from os.path import dirname, abspath

import numpy
from numpy.core.numeric import normalize_axis_tuple
sys.path.append(dirname(dirname(abspath(__file__))))
#from environments.gym_gomoku.envs import state
from environments.Gomoku import GomokuEnv
from collections import deque
from math import sqrt,log
import random


""" NODE CLASSES """
class Node():
    def __init__(self,state,parent_node=None,parent_action=None,terminal=False,legal_actions=None,content: dict = {}):
        self.state = state
        self.parent_node = parent_node
        self.parent_action = parent_action
        self.successors = []
        self.depth = 0 if parent_node is None else self.parent.depth + 1
        self.terminal = terminal
        self.all_legal_actions = legal_actions
        self.non_expanded_legal_actions = legal_actions
        self.content = content

 
    def find_successor_after_action(self,action):
        raise NotImplemented()

    def find_random_unexpanded_successor(self):
        random_idx = random.randint(0,len(self.non_expanded_legal_actions)-1)
        action = self.non_expanded_legal_actions[random_idx]
        return self.result(action)

    def find_rest_unexpanded_successors(self):
        generation = []
        for action in self.non_expanded_legal_actions:
            generation.append(self.result(action))
        return generation
        
    def append_successors_to_node(self,successors: list):
        for node in successors:
            self.successors.append(node)
            self.unappended_actions.remove(node.get_parent_action())

    def expand_random_successor(self):
        node = self.find_random_unexpanded_successor()
        self.append_successors_to_node([node])
        return node

    def expand_rest_successors(self):
        nodes = self.find_rest_unexpanded_successors()
        self.append_successors_to_node(nodes)
        return nodes

    """ Getters """
    def is_terminal(self):
        return self.terminal

    def is_completely_expanded(self):
        return len(self.unappended_actions)

    def is_root(self):
        return self.parent == None

    def get_depth(self):
        return self.depth

    def get_state(self):
        return self.state

    def get_content(self):
        return self.content

    def get_content_item(self,key):
        return self.content[key]

    def set_content_item(self,key,value):
        self.content[key] = value

    def get_successors(self):
        return self.children

    def get_all_legal_actions(self):
        if(self.all_legal_actions != None):
            self.all_legal_actions
        else:
            raise ValueError("Impossible to get all legal actions")

    def get_parent_node(self):
        return self.parent_node

    def get_parent_action(self):
        return self.parent_action



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

    
        
        

 
  
""""""""""""""""""""""""""""""""""""""       
        
class Search():
    def __init__(self,root,initial_blocks = [],initial_variables = {}):
        self.root = root
        self.current_node = self.root
        self.assembly_line = initial_blocks
        self.variables = initial_variables

    def run(self):
        block_no = 0
        while(len(self.assembly_line) != 0):
            self.assembly_line.pop().run(self)
            block_no += 1
        
class Block():
    def run(search):
        return NotImplemented() 

class Selection_Phase_Block(Block):
    def run(self,search):
        while search.current_node.is_completely_expanded():
            search.current_node = self.exploitation_exploration_criteria(search.current_node)

    def exploitation_exploration_criteria(self):
        return NotImplemented()


class Expansion_Phase_Block(Block):
    def run(self,search):
        if not search.current_node.is_terminal():
            search.current_node = self.expansion_criteria(search.current_node)
            self.set_node_predifined_contents(search.current_node)
        
    def expansion_criteria(self):
        return NotImplemented()


class Simulation_Phase_Block(Block):
    def run(self,search):
        while not search.current_node.is_terminal():
            search.current_node = self.fast_generation_policy(search.current_node)

    def fast_generation_policy(self,node):
        return NotImplemented()


class Backpropagation_Phase_Block(Block):
    def run(self,search):
        search.variables['last_node'] = search.current_node
        self.content_to_be_backpropagated = self.backpropagation_get_content_to_be_backpropagated()
        while not search.current_node.is_root():
            self.backpropagation_update_nodes(search.current_node,self.content_to_be_backpropagated,self.last_node)
            search.current_node = search.current_node.get_parent_node()

    def backpropagation_get_content_to_be_backpropagated(self,node):
        return NotImplemented()

    def backpropagation_update_nodes(self,node,last_node):
        return NotImplemented()

    def backpropagation_update_nodes(self,node):
        return NotImplemented()

class MCTS_Block(Block):
    def run(self,search):
        search.assembly.insert(0,Selection_Phase_Block())
        search.assembly.insert(0,Expansion_Phase_Block())
        search.assembly.insert(0,Simulation_Phase_Block())
        search.assembly.insert(0,Backpropagation_Phase_Block())
        search.assembly.insert(0,MCTS_Block())


print("MAIN")
search = Search(initial_blocks=[MCTS_Block()])
print("END MAIN")

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
