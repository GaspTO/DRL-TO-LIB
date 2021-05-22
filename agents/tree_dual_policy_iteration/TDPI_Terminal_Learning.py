from agents.tree_dual_policy_iteration.Tree_Dual_Policy_Iteration import Tree_Dual_Policy_Iteration, Config_Tree_Dual_Policy_Iteration
from time import time
import numpy as np
from collections import namedtuple
import torch
import random
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.trainer.trainer import Trainer




Episode_Tuple = namedtuple('Episode_Tuple',
    ['episode_observations',
    'episode_masks',
    'episode_actions',
    'episode_rewards',
    'episode_discounted_returns',
    'episode_next_observations',
    'episode_dones',])

State_Transition = namedtuple('State_Transition',
    ['observation',
    'mask',
    'action',
    'reward',
    'discounted_return',
    'next_observation',
    'done'])

Data = namedtuple('Data',
    ['observation',
    'discounted_return'])


class Config_TDPI_Terminal_Learning(Config_Tree_Dual_Policy_Iteration):
    def __init__(self,config=None):
        super().__init__(config)
        

class TDPI_Terminal_Learning(Tree_Dual_Policy_Iteration):
    agent_name = "TDPI_Terminal_Learning"
    def __init__(self,network,tree_agent,config):
        super().__init__(network,tree_agent,config)
        self.data_set = []

    def step(self):
        self.start = time()
        self.action = torch.tensor(self.tree_agent.play(self.observation)[0].argmax()) 
        self.next_observation, self.reward, self.done, _ = self.environment.step(self.action)

    def reset(self):
        super().reset()
        self.episode_state_values = []

    def save_step_info(self):
        super().save_step_info()
        if self.done == True:
            self.disc_returns = self.calculate_discounted_episode_returns(self.episode_rewards,discount_rate=1)
            for i in range(len(self.episode_observations)):
                self.data_set.append(Data(self.episode_observations[i],self.disc_returns[i]))
            self.data_set = self.data_set[-self.config.get_max_transition_memory():]

    def calculate_discounted_episode_returns(self,episode_rewards,discount_rate):
            discounted_returns = []
            discounted_total_reward = 0.
            for ix in range(len(episode_rewards)):
                discounted_total_reward = episode_rewards[-(ix + 1)] + discount_rate*discounted_total_reward
                discounted_returns.insert(0,np.array([discounted_total_reward]))
            return discounted_returns

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    *                            LEARNING METHODS     
    *                            Terminal Learning                                 
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def learn(self):
        dataloader = DataLoader(self.data_set,batch_size=self.config.get_batch_size(),shuffle=True,)
        trainer = Trainer(max_epochs=self.config.get_learn_epochs())
        trainer.fit(self.network,dataloader)


    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    *                            Other Methods...             
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def log_iteration_text(self,i,debug=False):
        modified_observation = str(self.episode_observations[i][0] + -1*self.episode_observations[i][1]) + "\n"
        reward_txt = "reward:{0: .2f} \n".format(self.episode_rewards[i])
        action_txt = "agent_action:{0: 2d} \n".format(self.episode_actions[i])
        return modified_observation + action_txt + reward_txt + "\n"
