from typing import AbstractSet
from agents.STI.Search_Interface import Search_Interface


class Basic_Search_Interface(Search_Interface):
    def __init__(self,state_space):
        super().__init(state_space)
        self.frontier_list = self.state_space.get_frontier_list()

    def step(self):
        



    



    



