from collections import Counter

import torch
import random
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from agents.Base_Agent import Base_Agent, Config_Base_Agent
from exploration_strategies.Epsilon_Greedy_Exploration import Epsilon_Greedy_Exploration
from utilities.data_structures.Replay_Buffer import Replay_Buffer



class Config_DQN(Config_Base_Agent):
    def __init__(self,config=None):
        Config_Base_Agent.__init__(self,config)
        if(isinstance(config,Config_DQN)):
            self.buffer_size = config.get_buffer_size()
            self.discount_rate = config.get_discount_rate()
            self.gradient_clipping_norm = config.get_gradient_clipping_norm()
            self.learning_iterations = config.get_learning_iterations()
            self.learning_rate = config.get_learning_rate()
            self.update_every_n_steps = config.get_update_every_n_steps()
        else:        
            self.buffer_size = 1000
            self.discount_rate = 0.99
            self.gradient_clipping_norm = 0.7
            self.learning_iterations = 1
            self.learning_rate = 0.1
            self.update_every_n_steps = 1

    def get_buffer_size(self):
        return self.buffer_size

    def get_discount_rate(self):
        return self.discount_rate

    def get_gradient_clipping_norm(self):
        return self.gradient_clipping_norm

    def get_learning_iterations(self):
        return self.learning_iterations

    def get_learning_rate(self):
        return self.learning_rate

    def get_update_every_n_steps(self):
        return self.update_every_n_steps    
    

        


class DQN(Base_Agent):
    """A deep Q learning agent"""
    agent_name = "DQN"
    def __init__(self, config):
        Base_Agent.__init__(self, config)
        self.memory = Replay_Buffer(self.config.get_buffer_size(), self.config.get_batch_size(), self.config.get_seed(), self.device)
        self.q_network_local = self.create_NN_through_NNbuilder(input_dim=self.input_shape, output_size=self.action_size)
        self.q_network_optimizer = optim.Adam(self.q_network_local.parameters(),
                                              lr=self.config.get_learning_rate(), eps=1e-4)
        self.exploration_strategy = Epsilon_Greedy_Exploration(config)


    """ Overloads """
    def reset_game(self):
        super(DQN, self).reset_game()
        self.update_learning_rate(self.config.get_learning_rate(), self.q_network_optimizer)

    def step(self):
        """Runs a step within a game including a learning step if required"""
        while not self.done:
            self.action = self.pick_action()
            self.conduct_action(self.action)
            if self.time_for_q_network_to_learn():
                for _ in range(self.config.get_learning_iterations()):
                    self.learn()
            self.save_experience()
            self.state = self.next_state #this is to set the state for the next iteration
            self.global_step_number += 1
        self.episode_number += 1


    """ Specific methods for the algorithm """
    def pick_action(self, state=None):

        """Uses the local Q network and an epsilon greedy policy to pick an action"""
        # PyTorch only accepts mini-batches and not single observations so we have to use unsqueeze to add
        # a "fake" dimension to make it a mini-batch rather than a single observation
        if state is None: state = self.state
        if isinstance(state, np.int64) or isinstance(state, int): state = np.array([state])
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        if len(state.shape) < 2: state = state.unsqueeze(0)
        self.q_network_local.eval() #puts network in evaluation mode
        with torch.no_grad():
            action_values = self.q_network_local(state)
        self.q_network_local.train() #puts network back in training mode
        if(self.action_mask_required == True):
            mask = self.get_action_mask()
            unormed_action_values =  action_values.mul(mask)
            action_values =  unormed_action_values/unormed_action_values.sum()
        else:
            mask = None
        action = self.exploration_strategy.perturb_action_for_exploration_purposes({"action_values": action_values,
                                                                                    "turn_off_exploration": self.turn_off_exploration,
                                                                                    "episode_number": self.episode_number,
                                                                                    "mask": mask})
        self.logger.info("Q values {} -- Action chosen {}".format(action_values, action))
        return action

    def learn(self, experiences=None):
        """Runs a learning iteration for the Q network"""
        if experiences is None: states, actions, rewards, next_states, dones = self.sample_experiences() #Sample experiences
        else: states, actions, rewards, next_states, dones = experiences
        self.loss = self.compute_loss(states, next_states, rewards, actions, dones)
        self.writer.add_scalar("lossloss",self.loss,self.global_step_number)
        actions_list = [action_X.item() for action_X in actions ]

        self.logger.info("Action counts {}".format(Counter(actions_list)))
        self.take_optimisation_step(self.q_network_optimizer, self.q_network_local, self.loss, self.config.get_gradient_clipping_norm())

    def compute_loss(self, states, next_states, rewards, actions, dones):
        """Computes the loss required to train the Q network"""
        with torch.no_grad():
            Q_targets = self.compute_q_targets(next_states, rewards, dones)
        Q_expected = self.compute_expected_q_values(states, actions)
        self.loss = F.mse_loss(Q_expected, Q_targets)
        return self.loss

    """ Network Auxiliary methods to the algorithm """
    def compute_q_targets(self, next_states, rewards, dones):
        """Computes the q_targets we will compare to predicted q values to create the loss to train the Q network"""
        Q_targets_next = self.compute_q_values_for_next_states(next_states)
        Q_targets = self.compute_q_values_for_current_states(rewards, Q_targets_next, dones)
        return Q_targets

    def compute_q_values_for_next_states(self, next_states):
        """Computes the q_values for next state we will use to create the loss to train the Q network"""
        Q_targets_next = self.q_network_local(next_states).detach().max(1)[0].unsqueeze(1)
        return Q_targets_next

    def compute_q_values_for_current_states(self, rewards, Q_targets_next, dones):
        """Computes the q_values for current state we will use to create the loss to train the Q network"""
        Q_targets_current = rewards + (self.config.get_discount_rate() * Q_targets_next * (1 - dones))
        return Q_targets_current

    def compute_expected_q_values(self, states, actions):
        """Computes the expected q_values we will use to create the loss to train the Q network"""
        Q_expected = self.q_network_local(states).gather(1, actions.long()) #must convert actions to long so can be used as index
        return Q_expected


    """ Other Auxiliary methods to the algorithm """
    def time_for_q_network_to_learn(self):
        """Returns boolean indicating whether enough steps have been taken for learning to begin and there are
        enough experiences in the replay buffer to learn from"""
        return self.right_amount_of_steps_taken() and self.enough_experiences_to_learn_from()

    def right_amount_of_steps_taken(self):
        """Returns boolean indicating whether enough steps have been taken for learning to begin"""
        return self.global_step_number % self.config.get_update_every_n_steps() == 0

    def sample_experiences(self):
        """Draws a random sample of experience from the memory buffer"""
        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences
        return states, actions, rewards, next_states, dones


    """ Other """
    def locally_save_policy(self):
        """Saves the policy"""
        torch.save(self.q_network_local.state_dict(), "Models/{}_local_network.pt".format(self.agent_name))

    
    