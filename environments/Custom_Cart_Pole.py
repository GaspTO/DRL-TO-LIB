from gym.envs.classic_control.cartpole import CartPoleEnv
from environments.core import Custom_Environment
import numpy as np

class Custom_Cart_Pole(Custom_Environment):
    def __init__(self):
        self.environment = CartPoleEnv()
        self.environment.reset()

    def step(self,action,observation=None):
        if observation is not None: raise ValueError("The Dynamic Observation environment setup isn't implemented")
        return self.environment.step(action)
        
    def reset(self):
        return self.environment.reset()

    def render(self,observation=None):
        if observation is not None: raise ValueError("The Dynamic Observation environment setup isn't implemented")
        return self.environment.render()

    def close(self):
        return self.environment.close()

    def seed(self,seed_n):
        return self.environment.seed(seed_n)

    def get_name(self) -> str:
        return "Cart-Pole"

    def needs_mask(self) -> bool:
        return False

    def get_mask(self,observation=None):
        if observation is not None: raise ValueError("The Dynamic Observation environment setup isn't implemented")
        return np.array([1,1]) #todo isn't suppose to return a mask

    def get_current_observation(self,observation=None,human=False):
        if observation is not None: raise ValueError("The Dynamic Observation environment setup isn't implemented")
        return np.array(self.environment.state)

    def get_legal_actions(self,observation=None):
        if observation is not None: raise ValueError("The Dynamic Observation environment setup isn't implemented")
        return np.array([1,1])

    def is_terminal(self, observation=None) -> bool:
        if observation is not None: raise ValueError("The Dynamic Observation environment setup isn't implemented")
        return self.environment.steps_beyond_done is not None #it's None when playing

    def get_winner(self, observation=None):
        raise ValueError("Cart-Pole is not an adversary game")

    def get_current_player(self,observation=None):
        raise ValueError("Cart-Pole is not an adversary game")

    def get_action_size(self):
        return self.environment.action_space.n #2
    
    def get_observation_shape(self) -> tuple:
        return self.environment.observation_space.shape

    
