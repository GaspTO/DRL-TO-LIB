from agents.policy_gradient_agents.REINFORCE import Config_Reinforce, REINFORCE
import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical
from agents.Base_Agent import Base_Agent, Config_Base_Agent


class Config_Reinforce_Baseline(Config_Reinforce):
    def __init__(self,config=None):
        Config_Reinforce.__init__(self,config)
        


class REINFORCE_Baseline(REINFORCE):
    agent_name = "REINFORCE_Baseline"
    def __init__(self, config):
        REINFORCE.__init__(self, config)
        self.policy = self.create_NN_through_NNbuilder(input_dim=self.input_shape, output_size=self.action_size + 1,smoothing=0.001)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config.get_learning_rate())
        
    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        super().reset_game()
        self.episode_log_probabilities = []
        self.episode_critic_values = []
        self.episode_step_number = 0

    def step(self):
        """Runs a step within a game including a learning step if required"""
        while not self.done:
            self.pick_and_conduct_action_and_save_log_probabilities()
            #self.update_next_state_reward_done_and_score()
            self.store_reward()
            if self.time_to_learn():
                self.actor_learn()
            self.state = self.next_state #this is to set the state for the next iteration
            self.episode_step_number += 1
        self.episode_number += 1

    """ Network -> Environment """
    def pick_and_conduct_action_and_save_log_probabilities(self):
        """Picks and then conducts actions. Then saves the log probabilities of the actions it conducted to be used for
        learning later"""
        action, log_probabilities, critic_value = self.pick_action_and_get_log_probabilities_and_critic_value()
        self.store_log_probabilities(log_probabilities)
        self.store_critic_value(critic_value)
        self.store_action(action)
        self.conduct_action(action)

    def pick_action_and_get_log_probabilities_and_critic_value(self):
        """Picks actions and then calculates the log probabilities of the actions it picked given the policy"""
        state = torch.from_numpy(self.state).float().unsqueeze(0).to(self.device)
        model_output = self.policy.forward(state).cpu() #todo questionable
        action_values = model_output[:, list(range(self.action_size))] #we only use first set of columns to decide action, last column is state-value
        critic_value = model_output[:, -1]
        print(critic_value)
        action_values_copy = action_values.detach()
        if(self.action_mask_required == True): #todo can't use the forward for this mask cause... critic_output
            mask = self.get_action_mask()
            unormed_action_values_copy =  action_values_copy.mul(mask)
            action_values_copy =  unormed_action_values_copy/unormed_action_values_copy.sum()
        

        action_distribution = Categorical(action_values_copy) # this creates a distribution to sample from
        action = action_distribution.sample()
        return action.item(), torch.log(action_values[0][action]), critic_value

    
    """ Learn """
    def actor_learn(self):
        """Runs a learning iteration for the policy"""
        total_discounted_reward = self.calculate_episode_discounted_reward()
        policy_loss = self.calculate_policy_loss_on_episode(total_discounted_reward)
        self.optimizer.zero_grad()
        policy_loss.backward()
        print("grad 81->" + str(self.policy.model[6].weight.grad[81]))
        self.optimizer.step()
        
        print("policy_loss = " + str(policy_loss.item()))

    def time_to_learn(self):
        """Tells us whether it is time for the algorithm to learn. With REINFORCE we only learn at the end of every
        episode so this just returns whether the episode is over"""
        return self.done

    
    """ Calculate Loss """

    def calculate_policy_loss_on_episode(self, total_discounted_reward):
        all_discounted_returns = torch.tensor(self.calculate_discounted_returns())
        all_critic_values = torch.cat(self.episode_critic_values)

        advantages = all_discounted_returns - all_critic_values 
        advantages = advantages.detach()

        critic_loss = (all_discounted_returns - all_critic_values)**2
        critic_loss = critic_loss.mean()

        action_log_probabilities_for_all_episodes = torch.cat(self.episode_log_probabilities)
        actor_loss = -1.0 * action_log_probabilities_for_all_episodes * advantages
        actor_loss = actor_loss.mean()
        
        total_loss = actor_loss + critic_loss
        return total_loss



        #return (-1.0 * torch.tensor(self.calculate_discounted_returns()) * torch.cat(self.episode_log_probabilities)).mean()
        """Calculates the loss from an episode"""
        """
        policy_loss = []
        for log_prob in self.episode_log_probabilities:
            policy_loss.append(-log_prob * total_discounted_reward)
        policy_loss = torch.cat(policy_loss).sum() # We need to add up the losses across the mini-batch to get 1 overall loss
        return policy_loss
        """

        
    def calculate_discounted_returns(self):
        discounted_returns = []
        discounted_reward = 0
        for ix in range(len(self.episode_rewards)):
            discounted_reward = self.episode_rewards[-(ix + 1)] + self.config.get_discount_rate()*discounted_reward
            discounted_returns.insert(0,discounted_reward)
        return discounted_returns

    def calculate_episode_discounted_reward(self): #this sucks
        """Calculates the cumulative discounted return for the episode"""
        discounts = self.config.get_discount_rate() ** np.arange(len(self.episode_rewards))
        total_discounted_reward = np.dot(discounts, self.episode_rewards)
        return total_discounted_reward

    def calculate_critic_loss_and_advantages(self, all_discounted_returns):
        """Calculates the critic's loss and the advantages"""
        critic_values = torch.cat(self.critic_outputs)
        advantages = torch.Tensor(all_discounted_returns) - critic_values
        advantages = advantages.detach()
        critic_loss =  (torch.Tensor(all_discounted_returns) - critic_values)**2
        critic_loss = critic_loss.mean()


    """ Storage """
    def store_log_probabilities(self, log_probabilities):
        """Stores the log probabilities of picked actions to be used for learning later"""
        self.episode_log_probabilities.append(log_probabilities)

    def store_critic_value(self,critic_value):
        self.episode_critic_values.append(critic_value)

    def store_action(self, action):
        """Stores the action picked"""
        self.action = action

    def store_reward(self):
        """Stores the reward picked"""
        self.episode_rewards.append(self.reward)


    
