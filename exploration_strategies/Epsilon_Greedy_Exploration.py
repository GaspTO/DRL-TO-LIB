from exploration_strategies.Base_Exploration_Strategy import Base_Exploration_Strategy
import numpy as np
import random
import torch

class Epsilon_Greedy_Exploration(Base_Exploration_Strategy):
    """Implements an epsilon greedy exploration strategy"""
    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon

    def perturb_action_for_exploration_purposes(self, action_vector: np.array, mask: np.array, info=None):
        if self.exploration:
            if random.random() > self.epsilon:
                return action_vector.argmax()
            else:
                idx_mask = np.where(mask == 1)[0]
                a = np.random.randint(0, len(idx_mask))
                return idx_mask[a]
        else:
            return action_vector.argmax()
 
    def reset(self):
        pass

       