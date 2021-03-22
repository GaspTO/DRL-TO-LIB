import numpy as np
''' This agent class is different from Base_Agent since
it does assume it's a learnable agent.
It can be a simple tree search'''

class Agent(): 
    def __init__(self,environment):
        self.environment = environment
        
    def play(self,state) -> int:
        '''
        output action number
        '''
        return NotImplementedError()



