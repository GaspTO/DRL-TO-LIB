from agents.STI.Search_Node import Search_Node
from agents.STI.Proxy_Node import Proxy_Node


class Environment_Search_Space():
    def __init__(self,observation=None):
        self.root = Search_Node(observation)
        self.current_node = self.root
        self.frontier_list = []
        self.closed_list = []

    def restart_iteration(self):
        self.current_iter_root = Proxy_Node(Proxy_Node(state=self.root),visited=True)
        self.current_node = self.root

    def set_current_node(self,node):
        self.current_node = node

    '''
    Proxy setters to handle the construction of the space
    '''
    def find_successor_after_action(self,node: Simple_Node,action):
        assert isinstance(node,Proxy_Node)
        node.get_state().find_successor_after_action(action)

    def find_random_unexpanded_successor(self,node: Simple_Node):
        return node.find_random_unexpanded_successor()
        
    def find_rest_unexpanded_successors(self,node):
        return node.find_rest_unexpanded_successors()

    def append_successors_to_node(self,node,successors:list):
        return node.append_successors_to_node(successors)
    
    def expand_random_successor(self,node):
        return node.expand_random_successor()

    def expand_rest_successors(self,node):
        return node.expand_rest_successors()

    '''
    Space getters
    '''
    def get_current_node(self):
        return self.current_node

    def get_root(self):
        return self.root

    def get_iteration_root(self):
        return self.current_iter_root

    def get_frontier_list(self):
        return self.frontier_list

    def get_closed_list(self):
        return self.closed_list
    

        


    


    

    


    




        
    

        


        

