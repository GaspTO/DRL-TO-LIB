import copy
import torch.nn as nn
import torch
from agents.Learning_Agent import Learning_Agent
from agents.DQN_agents.DQN import DQN, Config_DQN


class Config_DQN_With_Fixed_Q_Targets(Config_DQN):
    def __init__(self,config=None):
        Config_DQN.__init__(self,config),
        if(isinstance(config,Config_DQN_With_Fixed_Q_Targets)):
            self.tau = config.get_tau()
            self.reset_every_n_episodes = config.get_reset_every_n_episodes()
        else:
            self.tau = 0.01
            self.reset_every_n_episodes = 1

    def get_tau(self):
        if(self.tau != None):
            return self.tau
        else:
            raise ValueError("Tau Not Defined")

    def get_reset_every_n_episodes(self):
        if(self.reset_every_n_episodes != None):
            return self.reset_every_n_episodes
        else:
            raise ValueError("Reset Every n Episodes Not Defined")


class DQN_With_Fixed_Q_Targets(DQN):
    """A DQN agent that uses an older version of the q_network as the target network"""
    agent_name = "DQN with Fixed Q Targets"
    def __init__(self, config):
        DQN.__init__(self, config)
        self.q_network_target = self.config.architecture()
        Learning_Agent.copy_model_over(from_model=self.q_network_local, to_model=self.q_network_target)

    def learn(self, transitions=None):
        super(DQN_With_Fixed_Q_Targets, self).learn(transitions=transitions)
        if(self.time_for_q_network_target_to_reset()):
            self.soft_update_of_target_network(self.q_network_local, self.q_network_target, self.config.get_tau())  

    def compute_q_values_for_next_states(self, next_states):
        """max_a Q^(t)(s^t+1,a)"""
        Q_targets_next = self.q_network_target(next_states).detach().max(1)[0].unsqueeze(1)
        return Q_targets_next

    def time_for_q_network_target_to_reset(self):
        return self.episode_number % self.config.get_reset_every_n_episodes() == 0