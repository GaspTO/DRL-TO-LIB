from abc import abstractmethod
from typing import AbstractSet


class Search_Interface:
    def __init__(self,state_space):
        self.state_space = state_space

    def execute(self,stop_condition):
        while not stop_condition():
            self.step()
        
    @abstractmethod
    def step(self):
        pass

    def get_frontier(self):
        return self.frontier_list

    def get_closed(self):
        return self.closed_list

    def is_frontier_empty(self):
        return len(self.frontier_list) == 0  

    def get_frontier_size(self):
        return len(self.frontier_list)

    def add_to_frontier(self,node):
        pass

    def get_next_node_in_frontier(self):
        pass     


