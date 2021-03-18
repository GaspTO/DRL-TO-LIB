import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from agents.Base_Agent import Base_Agent, Config_Base_Agent
from agents.policy_gradient_agents.REINFORCE import REINFORCE, Config_Reinforce
from algorithms.Search import MCTS_Search, K_Row_MCTSNode



class Config_Reinforce_adv_krow(Config_Reinforce):
    def __init__(self,config=None):
        Config_Base_Agent.__init__(self,config)


class REINFORCEadv_krow(REINFORCE):
    agent_name = "REINFORCEadv"
    def __init__(self, config):
        REINFORCE.__init__(self, config)
        if(self.get_environment_title() != 'K_Row'): raise ValueError("This algorithm only supports the K_ROW game")
 
    def conduct_action(self):
        action1, log_prob1 = self.pick_action_and_get_log_probabilities()
        next_state, reward, done, _ = self.environment.step(action1)
        reward2 = 0
        if done == False:
            search = MCTS_Search(K_Row_MCTSNode(self.environment.state))
            search.run_n_playouts(25)
            action = search.play_action()
            next_state, reward2, done, _ = self.environment.step(action)
        self.action = action1
        self.reward = reward + reward2
        self.done = done
        self.state = next_state
        self.episode_actions.append(self.action)
        self.episode_rewards.append(self.reward)
        self.episode_dones.append(self.done)
        self.episode_states.append(self.state)
        self.episode_action_log_probabilities.append(log_prob1)
        self.total_episode_score_so_far += self.reward
        if self.config.get_clip_rewards(): self.reward = max(min(self.get_reward, 1.0), -1.0)
        if(self.done == True):
            self.logger.info("final_reward: {}".format(self.reward))

