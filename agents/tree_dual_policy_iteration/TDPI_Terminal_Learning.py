from agents.tree_dual_policy_iteration.Tree_Dual_Policy_Iteration import Tree_Dual_Policy_Iteration
from time import time
import numpy as np
from collections import namedtuple
import torch.optim as optim
import torch
import random

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


class TDPI_Terminal_Learning(Tree_Dual_Policy_Iteration):
    agent_name = "TDPI_Terminal_Learning"
    def __init__(self,config):
        super().__init__(config)
        
    def step(self):
        self.start = time()
        self.action, info = self.k_best_first_minimax_expert(self.observation,k=2,iterations=50)
        self.state_value = info["state_value"]   
        self.next_observation, self.reward, self.done, _ = self.environment.step(self.action)

    def reset(self):
        super().reset()
        self.episode_state_values = []

    def save_step_info(self):
        super().save_step_info()
        #* state value
        self.episode_state_values.append(self.state_value)
        #* save trajectories
        if self.done == True:
            episode = Episode_Tuple(
                self.episode_observations,
                self.episode_masks,
                self.episode_actions,
                self.episode_rewards,
                self.calculate_discounted_episode_returns(self.episode_rewards,discount_rate=1),
                self.episode_next_observations,
                self.episode_dones)
            for transition in zip(*episode):
                self.add_transition_to_memory(State_Transition(*transition),self.max_transition_memory)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    *                            LEARNING METHODS     
    *                       Learning on Trajectories                                 
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def learn(self):
        transitions_to_train = self.get_transition_batch(samples=self.num_transitions_to_sample)
        transition_batch = []
        for _ in range(self.learn_epochs):
            #! use split
            for transition in transitions_to_train:
                transition_batch.append(transition)
                if len(transition_batch) == self.batch_size:
                    total_loss = self.loss_value_on_trajectory(transition_batch)
                    self.take_optimisation_step(self.optimizer,self.network,total_loss, self.config.get_gradient_clipping_norm())
                    transition_batch = []

    def loss_value_on_trajectory(self,transition_batch):
        observations = np.array([item.observation for item in transition_batch])
        discounted_returns = torch.tensor([item.discounted_return for item in transition_batch])
        state_values = self.network.load_observations(observations).get_state_value()
        loss_vector = (state_values - discounted_returns)**2
        loss = loss_vector.mean()
        return loss

    def calculate_discounted_episode_returns(self,episode_rewards,discount_rate):
            discounted_returns = []
            discounted_total_reward = 0.
            for ix in range(len(episode_rewards)):
                discounted_total_reward = episode_rewards[-(ix + 1)] + discount_rate*discounted_total_reward
                discounted_returns.insert(0,discounted_total_reward)
            return discounted_returns

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    *                            Other Methods...             
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def log_iteration_text(self,i,debug=False):
        modified_observation = str(self.episode_observations[i][0] + -1*self.episode_observations[i][1]) + "\n"
        reward_txt = "reward:{0: .2f} \n".format(self.episode_rewards[i])
        action_txt = "agent_action:{0: 2d} \n".format(self.episode_actions[i])
        return modified_observation + action_txt + reward_txt + "\n"
