from abc import abstractmethod
from os import environ
import numpy as np

'''
this agent class is different from Base_Agent since it does assume it's a learnable agent.
An Agent only makes sense in the context of an environment
'''
class Agent(): 
    def __init__(self,environment):
        self.environment = environment
        self.observation = self.environment.get_current_observation()
        self.next_observation = None
        self.action = None
        self.reward = None
        self.done = False
    
    '''
    Returns a tuple with:
        - np.array with actions for each observation
        - dict of extra info
        - if observations are passed, then uses them. otherwise, uses internal environment state
        - play DOES NOT change the state of the environment.
    '''
    @abstractmethod
    def play(self,observations:np.ndarray=None,policy=None,info=None) -> tuple([np.array,dict]):
        return NotImplementedError

    def get_environment(self):
        return self.environment

    def reset(self):
        self.observation = self.environment.reset()
        self.mask = self.environment.get_mask()
        self.next_observation = None
        self.action = None
        self.reward = None
        self.done = False

    





