import numpy as np
''' This agent class is different from Base_Agent since
it does assume it's a learnable agent.
It can be a simple tree search'''

class Agent(): 
    def __init__(self,environment):
        self.environment = environment
        
    def play(self,info) -> int:
        '''
        1ª builds environment from info
        2ª does whatever it needs. Maybe only get the observation. Maybe it's an expert system that will need more...
        '''
        return NotImplementedError()



