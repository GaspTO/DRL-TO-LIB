from agents.tree_dual_policy_iteration.Tree_Dual_Policy_Iteration import Tree_Dual_Policy_Iteration, Config_Tree_Dual_Policy_Iteration
from time import time
import numpy as np
from collections import namedtuple
import torch
import random
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.trainer.trainer import Trainer
from environments.Arena import Arena
from agents.tree_agents.MCTS_Search import MCTS_Search
from exploration_strategies import Epsilon_Greedy_Exploration




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
        self.use_arena = False

        
    def play(self,observations:np.array=None,info=None) -> tuple([np.array,dict]):
        if observations is None:
            observations = np.array([self.environment.get_current_observation()])
        actions, info = self.tree_agent.play(observations)
        return actions, info

    def reset(self):
        super().reset()

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
        if self.use_arena:
            previous_network = deepcopy(self.network)
            previous_tree_agent = deepcopy(self.tree_agent)
            previous_tree_agent.set_network(previous_network)
            previous_play = lambda obs: previous_tree_agent.play(obs)[0].argmax()
        
        num_of_samples = (self.config.get_batches_per_epoch()*self.config.get_batch_size())
        if num_of_samples > len(self.data_set):
            data = self.data_set
        else:
            data = random.sample(self.data_set,num_of_samples)

        dataloader = DataLoader(data,batch_size=self.config.get_batch_size(),shuffle=True,num_workers=self.config.get_num_data_workers())
        trainer = Trainer(max_epochs=self.config.get_learn_epochs(),checkpoint_callback=False)
        trainer.fit(self.network,dataloader)

        if self.use_arena:
            new_play = lambda obs: self.tree_agent.play(obs)[0].argmax()
            wins = Arena(self.environment.environment).playGames(previous_play,new_play,10)
            if wins[0] > wins[1]:
                self.network = previous_network
        
        if self.episode_number % 300 == 0:
            self.test(30)


    def test(self,n):
        self.tree_agent.exploration_st.turn_off_exploration()
        mcts = MCTS_Search(self.environment.environment,n_iterations=100)
        net_play = lambda obs: self.tree_agent.play(np.array([obs]))[0][0]
        mcts_play = lambda obs: mcts.play(obs)
        wins = Arena(self.environment.environment).playGames(net_play,mcts_play,n)
        self.tree_agent.exploration_st.turn_on_exploration()
        self.logger.info("Test Results:\n" +  "agent: " + str(wins[0]) + " mcts:" + str(wins[1]) )


    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    *                            Other Methods...             
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def log_iteration_text(self,i,debug=False):
        modified_observation = str(self.episode_observations[i][0] + -1*self.episode_observations[i][1]) + "\n"
        reward_txt = "reward:{0: .2f} \n".format(self.episode_rewards[i])
        action_txt = "agent_action:{0: 2d} \n".format(self.episode_actions[i])
        return modified_observation + action_txt + reward_txt + "\n"
