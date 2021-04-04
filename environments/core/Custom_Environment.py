import gym
import numpy as np
from environments.core.Players import Players, Player


'''
This environment is an adapter necessary to run the algorithms in this library
Any environment that needs to be used needs to be adapted through this interface

The environments keep an internal state, which is used every time no observation is passed
in the arguments. When an observation is passed, the internal state is kept, but ignored, 
and the environment executes its functions based on the observation passed. (This observation
does not update the internal state)
'''
class Custom_Environment(gym.Env):
    '''
    gym env interface
    '''
    def step(self,action,observation=None) -> tuple:
        raise NotImplementedError
        
    def reset(self) -> np.ndarray:
        raise NotImplementedError

    def render(self,observation=None):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def seed(self,seed_n):
        raise NotImplementedError

    '''
    interesting getters
    '''
    def get_name(self) -> str:
        raise NotImplementedError

    def needs_mask(self) -> bool:
        raise NotImplementedError

    def get_mask(self,observation=None) -> np.ndarray:
        raise NotImplementedError

    def get_current_observation(self,observation=None,human=False) -> np.ndarray:
        raise NotImplementedError

    def get_legal_actions(self,observation=None) -> np.ndarray:
        raise NotImplementedError

    def is_terminal(self, observation=None) -> bool:
        raise NotImplementedError

    def get_winner(self, observation=None):
        raise NotImplementedError

    def get_current_player(self,observation=None) -> Player:
        raise NotImplementedError

    def get_action_size(self) -> int:
        raise NotImplementedError
    
    def get_observation_shape(self) -> tuple:
        raise NotImplementedError

    '''set state'''
    def set_current_state(self,observation):
        raise NotImplementedError
    


    
    

