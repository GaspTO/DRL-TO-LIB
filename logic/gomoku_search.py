import os
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
#from environments.gym_gomoku.envs import state
from environments.Gomoku import GomokuEnv
from collections import deque




class Node():
    def __init__(self,state,parent_node=None):
        self.state = state
        self.parent = parent_node
        self.successors = []
        if(parent_node is None):
            self.depth = 0
        else:
            self.depth = self.parent.depth + 1

    def add_child_node(self,child):
        self.successos.append(child)

    def add_children_nodes(self,children):
        for child in children:
            self.successors.append(child)
                 


class Gomoku_Tree():
    def __init__(self,state):
        self.root = Node(state)
     
    def generate_children_nodes_of_node(self,node):
        children = []
        for action in node.state.board.get_legal_action():
            new_node = Node(node.state.act(action),node)
            children.append(new_node)
        node.add_children_nodes(children)
        return children

    def BFS(self,max_depth=-1):
        stack = deque()
        self.current_node = self.root
        nodes_expanded = 1
        solution_found = False
        while(solution_found == False and self.current_node.depth < max_depth): #< cause we're looking at children
            for child in self.generate_children_nodes_of_node(self.current_node): #todo we can make this so much more efficient
                if(child.state.board.is_terminal()):
                    return child.state.board.last_action
                stack.appendleft(child)
            self.current_node = stack.pop()
            nodes_expanded += 1
        return None
    
    '''
    def DFS(self,state,max_level=-1):
        #todo incomplete
        stack = deque()
        stack.append(state)
        solution_found = False
        while(solution_found == False):
            state = stack.pop()
            if(state.board.is_terminal()):
                return state
            else:
                for child in generate_children(state): #todo we can make this so much more efficient
                    stack.append(child)
    '''

    
        


'''
env = GomokuEnv('black','random',9)
init_state = env.state
tree = Gomoku_Tree(init_state)
result = tree.BFS(3)
print(result)
'''






