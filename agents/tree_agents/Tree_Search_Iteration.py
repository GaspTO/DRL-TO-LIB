from abc import abstractmethod, abstractproperty
from agents.Agent import Agent

class Tree_Search_Iteration(Agent):
    def __init__(self,environment):
        super().__init__(environment)

    '''
    play
        chooses action for observation
    '''
    @abstractmethod
    def play(self,observation = None):
        pass

    '''
    do_search_phase
        runs a search algorithm until some condition is met
    '''
    @abstractmethod
    def do_search_phase(self):
        pass

    '''
    do_backtrack_phase
        uses the the current search tree to update its nodes with values that will aid
        the future searches. 
    '''
    @abstractmethod
    def do_backtrack_phase(self):
        pass

    
    

