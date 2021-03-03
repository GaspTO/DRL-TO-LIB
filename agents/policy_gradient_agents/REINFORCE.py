import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical
from agents.Base_Agent import Base_Agent, Config_Base_Agent


class Config_Reinforce(Config_Base_Agent):
    def __init__(self,config=None):
        Config_Base_Agent.__init__(self,config)
        if(isinstance(config,Config_Reinforce)):
            self.discount_rate = config.get_discount_rate()
            self.learning_rate = config.get_learning_rate()
        else:
            self.discount_rate = 0.99
            self.learning_rate = 1
    
    def get_discount_rate(self):
        if(self.discount_rate == None):
            raise ValueError("Discount Rate Not Defined")
        return self.discount_rate

    def get_learning_rate(self):
        if(self.learning_rate == None):
            raise ValueError("Learning Rate Not Defined")
        return self.learning_rate


class REINFORCE(Base_Agent):
    agent_name = "REINFORCE"
    def __init__(self, config):
        Base_Agent.__init__(self, config)
        self.policy = self.create_NN_through_NNbuilder(input_dim=self.input_shape, output_size=self.action_size,smoothing=0.001)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config.get_learning_rate())

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        super().reset_game()
        self.episode_action_log_probabilities = []
        self.episode_step_number = 0

    def step(self):
        """Runs a step within a game including a learning step if required"""
        while not self.done:
            self.pick_and_conduct_action_and_save_log_probabilities()
            if self.time_to_learn():
                self.actor_learn()
            self.state = self.next_state #this is to set the state for the next iteration
            self.episode_step_number += 1
        self.episode_number += 1

    def pick_and_conduct_action_and_save_log_probabilities(self):
        """Picks and then conducts actions. Then saves the log probabilities of the actions it conducted to be used for
        learning later"""
        self.action, log_probabilities = self.pick_action_and_get_log_probabilities()
        self.store_log_probabilities(log_probabilities)
        self.conduct_action(self.action)

    def pick_action_and_get_log_probabilities(self):
        """Picks actions and then calculates the log probabilities of the actions it picked given the policy"""
        # PyTorch only accepts mini-batches and not individual observations so we have to add
        # a "fake" dimension to our observation using unsqueeze
        state = torch.from_numpy(self.state).float().unsqueeze(0).to(self.device)
        action_values = self.policy.forward(state,self.get_action_mask()).cpu() 
        action_values_copy = action_values.detach()
        action_distribution = Categorical(action_values_copy) # this creates a distribution to sample from
        action = action_distribution.sample()
        return action.item(), torch.log(action_values[0][action])

    
    """ Learn """
    def actor_learn(self):
        """Runs a learning iteration for the policy"""
        total_discounted_reward = self.calculate_episode_discounted_reward()
        policy_loss = self.calculate_policy_loss_on_episode(total_discounted_reward)
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        print("...")

    def time_to_learn(self):
        """Tells us whether it is time for the algorithm to learn. With REINFORCE we only learn at the end of every
        episode so this just returns whether the episode is over"""
        return self.done

    
    """ Calculate Loss """
    def calculate_discounted_returns(self):
        discounted_returns = []
        discounted_reward = 0
        for ix in range(len(self.episode_rewards)):
            discounted_reward = self.episode_rewards[-(ix + 1)] + self.config.get_discount_rate()*discounted_reward
            discounted_returns.insert(0,discounted_reward)
        return discounted_returns

    def calculate_episode_discounted_reward(self):
        """Calculates the cumulative discounted return for the episode"""
        discounts = self.config.get_discount_rate() ** np.arange(len(self.episode_rewards))
        total_discounted_reward = np.dot(discounts, self.episode_rewards)
        return total_discounted_reward

    def calculate_policy_loss_on_episode(self, total_discounted_reward):

        #return (-1.0 * torch.tensor(self.calculate_discounted_returns()) * torch.cat(self.episode_log_probabilities)).mean()
        """Calculates the loss from an episode"""
        
        policy_loss = []
        for log_prob in self.episode_log_probabilities:
            policy_loss.append(-log_prob * total_discounted_reward)
        policy_loss = torch.cat(policy_loss).sum() # We need to add up the losses across the mini-batch to get 1 overall loss
        return policy_loss
        

    """ store in list """
    def store_action_log_probability(self, action_log_probability):
        """Stores the log probability of picked actions to be used for learning later"""
        self.episode_log_probabilities.append(action_log_probability)

    """ get in lists """
    def get_episode_action_log_probabilities(self):
        return self.episode_log_probabilities


    
