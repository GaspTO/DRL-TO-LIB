import os
import sys
from os.path import dirname, abspath

from numpy.core.numeric import normalize_axis_tuple
sys.path.append(dirname(dirname(abspath(__file__))))
#from environments.gym_gomoku.envs import state
from environments.Gomoku import GomokuEnv
from collections import deque
from math import sqrt,log
import random


""" NODE CLASSES """
class Node():
    def __init__(self,state,parent_node=None,mother_action=None):
        self.state = state
        self.parent = parent_node
        self.mother_action = mother_action
        self.children = []
        if(parent_node is None):
            self.depth = 0
        else:
            self.depth = self.parent.depth + 1
        
        if(self.state.board.is_terminal()): #if terminal, set winner
            self.value = 1 if self.state.color == self.state.board.winner else -1
            self.expanded = True
            self.terminal = True
        else:
            self.value = 0
            self.expanded = False
            self.terminal = False

        self.non_expanded_legal_actions = self.state.board.get_legal_action()
        
    
    """ Generation Methods independent from expansion """
    def generate_all_children(self):
        return [self.new_node(self.transition(action),action) for action in self.state.board.get_legal_action()]

    def generate_any_child(self):
        all_legal_actions = self.state.board.get_legal_action()
        idx = random.randint(0,len(all_legal_actions)-1)
        action = all_legal_actions[idx]
        child = self.new_node(self.transition(action),action)
        return child

    """ Generation Methods dependent from expansion """
    def generate_rest_of_children(self, erase_actions = True):
        children = []
        for action in self.non_expanded_legal_actions:
            new_node = self.new_node(self.transition(action),action)
            children.append(new_node)
        if(erase_actions):
            self.non_expanded_legal_actions = []
        return children


    def generate_a_non_expanded_child(self, erase_action = True):
        if(self.is_expanded()): raise ValueError("Node is already expanded")
        idx = random.randint(0,len(self.non_expanded_legal_actions)-1)
        if(erase_action):
            action = self.non_expanded_legal_actions.pop(idx)
        else:
            action = self.non_expanded_legal_actionss[idx]
        child = self.new_node(self.transition(action),action)
        return child

    
    """ Expansion - Generation + Appending """
    def expand(self):
        if(self.is_expanded()): raise ValueError("Node already expanded")
        else:
            children = self.generate_rest_of_children(erase_actions = True)
            if(self.children == []): self.children = children
            else: self.children.extend(children)
        self.expanded = True
        return self.children

    def expand_one_child(self):
        if(self.is_expanded()): raise ValueError("Node already expanded, can't add another child")  
        else:
            child = self.generate_a_non_expanded_child(erase_action = True)
            self.children.append(child)
        if(len(self.non_expanded_legal_actions) == 0):
            self.expanded = True
        return child
        
    def is_terminal(self):
        return self.terminal

    def is_expanded(self):
        return self.expanded

    def is_root(self):
        return self.parent == None
    
    def get_children(self):
        return self.children

    def transition(self,action):
        return self.state.act(action)

    def new_node(self,state,mother_action):
        return Node(state,self,mother_action)

class MCTS_Node(Node):
    def __init__(self,env,state,parent_node=None,mother_action=None):
        Node.__init__(self,state,parent_node,mother_action)
        self.env = env
        self.num_wins = 0
        self.num_visits = 0
        self.belongs_to_tree = True
        
    def new_node(self,state,action):
        return MCTS_Node(env,state,self,action)
    
    def transition(self, action): #todo this is because of simulator
        prev_state = self.state
        state = super().transition(action)
        if(state.board.is_terminal()):
            return state # when I win I don't want my opponent to play
        next_state =  env.exec_opponent_play(state, prev_state, action)[0]
        return next_state
    
""" ALGORITHMS ON GOMOKU """
class Basic_Gomoku_Search():
    def __init__(self,state):
        self.root = Node(state)
     
    def BFS(self,max_depth=-1):
        i = 0
        stack = deque()
        self.current_node = self.root
        nodes_expanded = 1
        solution_found = False
        while(solution_found == False and self.current_node.depth < max_depth): #< cause we're looking at children
            print("i=" + str(i))
            i += 1
            self.current_node.expand()
            for child in self.current_node.get_children():
                if(child.is_terminal()):
                    return child.state.board.last_action
                stack.appendleft(child)
            self.current_node = stack.pop()
            nodes_expanded += 1
        return None

    def DFS(self,max_depth=-1):
        i = 0
        stack = deque()
        self.current_node = self.root
        nodes_expanded = 1
        solution_found = False
        while(solution_found == False and (self.current_node.depth < max_depth or max_depth == -1)): #< cause we're looking at children
            print("i=" + str(i))
            i += 1
            self.current_node.expand()
            for child in self.current_node.get_children():
                if(child.is_terminal()):
                    return child.state.board.last_action
                stack.append(child)
            self.current_node = stack.pop()
            nodes_expanded += 1
        return None
    

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
