from collections import Counter

import torch
import torch.nn as nn
import random
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from agents.Learning_Agent import Learning_Agent, Config_Learning_Agent
from exploration_strategies.Epsilon_Greedy_Exploration import Epsilon_Greedy_Exploration
from utilities.data_structures.Replay_Buffer import Replay_Buffer



class Config_DQN(Config_Learning_Agent):
    def __init__(self,config=None):
        Config_Learning_Agent.__init__(self,config)
        if(isinstance(config,Config_DQN)):
            self.buffer_size = config.get_buffer_size()
            self.discount_rate = config.get_discount_rate()
            self.learning_iterations = config.get_learning_iterations()
            self.learning_rate = config.get_learning_rate()
            self.update_every_n_steps = config.get_update_every_n_steps()
        else:        
            self.buffer_size = 1000
            self.discount_rate = 0.99
            self.learning_iterations = 1
            self.learning_rate = 0.1
            self.update_every_n_steps = 1

    def get_buffer_size(self):
        return self.buffer_size

    def get_discount_rate(self):
        return self.discount_rate

    def get_learning_iterations(self):
        return self.learning_iterations

    def get_learning_rate(self):
        return self.learning_rate

    def get_update_every_n_steps(self):
        return self.update_every_n_steps    
    

  

class DQN(Learning_Agent):
    """A deep Q learning agent"""
    agent_name = "DQN"
    def __init__(self, config):
        Learning_Agent.__init__(self, config)
        self.memory = Replay_Buffer(self.config.get_buffer_size(), self.config.get_batch_size(), self.config.get_seed(), self.device)
        self.q_network_local = self.config.architecture()
        self.q_network_optimizer = optim.Adam(self.q_network_local.parameters(),lr=self.config.learning_rate)
        self.exploration_strategy = Epsilon_Greedy_Exploration(config)


    """ Basic Operations """
    def reset_game(self):
        super(DQN, self).reset_game()
        self.update_learning_rate(self.config.get_learning_rate(), self.q_network_optimizer)

    
    def step(self):
        """Runs a step within a game including a learning step if required"""
        while not self.done:
            self.conduct_action()
            if self.time_for_q_network_to_learn():
                for _ in range(self.config.get_learning_iterations()):
                    self.learn()
            self.global_step_number += 1
        self.episode_number += 1
    
    def step(self):
        action = self.pick_action()
        next_state, reward, done, _ = self.environment.step(action)
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.save_transition(transition=(self.state,self.action,self.reward,self.next_state,self.done))
        self.state = self.next_state #only update state after saving transition
        self.episode_states.append(self.state)
        self.episode_actions.append(action)
        self.episode_rewards.append(self.reward)
        self.total_episode_score_so_far += self.reward
        if self.config.get_clip_rewards(): self.reward =  max(min(self.reward, 1.0), -1.0)
        if(self.done == True):
            self.logger.info("final_reward: {}".format(self.reward))


    def pick_action(self):
        """Uses the local Q network and an epsilon greedy policy to pick an action"""
        state = torch.from_numpy(self.state).float().unsqueeze(0).to(self.device)
        self.q_network_local.eval() #puts network in evaluation mode
        with torch.no_grad():
            action_values = self.q_network_local(state)
        true_action_values = action_values.clone()
        if(self.action_mask_required == True): 
            mask = self.get_action_mask()
            unormed_action_values_copy =  action_values.mul(mask)
            action_values =  unormed_action_values_copy/unormed_action_values_copy.sum()
        self.q_network_local.train()
        action = self.exploration_strategy.perturb_action_for_exploration_purposes({"action_values": action_values,
                                                                                    "turn_off_exploration": self.turn_off_exploration,
                                                                                    "episode_number": self.episode_number,
                                                                                    "mask": self.get_action_mask()})
        if(self.debug_mode): self.logger.info("Q values\n {} -- Action chosen {} Masked_Prob {} True_Prob {}".format(action_values, action,action_values[0][action],true_action_values[0][action]))
        else: self.logger.info("Action chosen {} Masked_Prob {} True_Prob {}".format(action,action_values[0][action],true_action_values[0][action]))
        return action

       
    """ Learn """
    def learn(self, transitions=None):
        """Runs a learning iteration for the Q network"""
        if(transitions is None):
            transitions = self.sample_transitions()
        states, actions, rewards, next_states, dones = transitions
        self.loss = self.compute_loss(states, next_states, rewards, actions, dones)
        self.take_optimisation_step(self.q_network_optimizer, self.q_network_local, self.loss, self.config.get_gradient_clipping_norm())
        actions_list = [action_X.item() for action_X in actions]
        if(self.debug_mode): self.logger.info("Batch to learn from: Action:nºtimes {}".format(Counter(actions_list)))

    def time_to_learn(self):
        """ official method of interface """
        return self.time_for_q_network_to_learn()

    def time_for_q_network_to_learn(self):
        """Returns boolean indicating whether enough steps have been taken for learning to begin and there are
        enough transitions in the replay buffer to learn from"""
        return self.right_amount_of_steps_taken() and self.enough_transitions_to_learn_from()

    def right_amount_of_steps_taken(self):
        """Returns boolean indicating whether enough steps have been taken for learning to begin"""
        return self.global_step_number % self.config.get_update_every_n_steps() == 0


    """ Calculate Loss """
    def compute_loss(self, states, next_states, rewards, actions, dones):
        """r + gamma * max_a Q(s^t+1,a) - Q(s^t,a^t) """
        Q_expected = self.compute_expected_q_values(states, actions)
        with torch.no_grad():
            Q_targets = self.compute_q_targets(next_states, rewards, dones)
        self.loss = F.mse_loss(Q_expected, Q_targets)
        return self.loss

    def compute_expected_q_values(self, states, actions):
        """ Q(s^t,a^t) """
        Q_expected = self.q_network_local(states).gather(1, actions.long()) #must convert actions to long so can be used as index
        return Q_expected

    def compute_q_targets(self, next_states, rewards, dones):
        """ Q_target(s^t,a^t) = r + gamma * max_a Q(s^t+1,a) """
        Q_targets_next = self.compute_q_values_for_next_states(next_states) 
        Q_targets = self.compute_q_values_for_current_states(rewards, Q_targets_next, dones)
        return Q_targets

    def compute_q_values_for_next_states(self, next_states):
        """max_a Q^(t)(s^t+1,a)"""
        Q_targets_next = self.q_network_local(next_states).detach().max(1)[0].unsqueeze(1)
        return Q_targets_next

    def compute_q_values_for_current_states(self, rewards, Q_targets_next, dones):
        """Q^(t+1)(s^t,a^t) =  r + gamma * [max_a Q^(t)(s^t+1,a)] """
        Q_targets_current = rewards + (self.config.get_discount_rate() * Q_targets_next * (1 - dones))
        return Q_targets_current



    
    