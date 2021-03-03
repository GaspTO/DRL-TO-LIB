from agents.policy_gradient_agents.REINFORCE import Config_Reinforce, REINFORCE
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions import Categorical
from agents.Base_Agent import Base_Agent, Config_Base_Agent
import pytorch_lightning as lightning
import math


class Config_Reinforce_Baseline(Config_Reinforce):
    def __init__(self,config=None):
        Config_Reinforce.__init__(self,config)
        

class Policy_ReBa(nn.Module):
    def __init__(self):
        super().__init__()
        self.one = nn.Sequential(
            nn.Linear(81, 30),
            nn.Sigmoid(),
            nn.Linear(30,30),
            nn.Sigmoid(),
            nn.Linear(30,30),
            nn.Sigmoid()
        )
        self.actions =  nn.Sequential(
            nn.Linear(30,81),
            #todo 
        ) 
        self.critic = nn.Sequential(
            nn.Linear(30,1)
        )

    def forward(self, x, smoothing = 0.001):
        # in lightning, forward defines the prediction/inference actions
        self.x1 = x.view(x.size(0),-1)
        self.x2 = self.one(self.x1)


        self.actions1 = self.actions(self.x2)
        self.actions2 = torch.softmax(self.actions1,dim=1)
        #self.actions3 = (self.actions2 + smoothing) 
        #self.actions4 = self.actions3/self.actions3.sum()

 
        self.x2.retain_grad()
        #self.actions1.retain_grad()
        self.actions2.retain_grad()
        #self.actions3.retain_grad()
        #self.actions4.retain_grad()
        
        critic = self.critic(self.x2)
        return self.actions2,critic

    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters(), lr=2e-13)
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-6)
        return optimizer
    

class REINFORCE_Baseline(REINFORCE):
    agent_name = "REINFORCE_Baseline"
    def __init__(self, config):
        REINFORCE.__init__(self, config)
        #self.policy = self.create_NN_through_NNbuilder(input_dim=self.input_shape, output_size=self.action_size + 1,smoothing=0.001)
        self.policy = Policy_ReBa()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config.get_learning_rate())
        
    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        super().reset_game()
        self.episode_log_probabilities = []
        self.episode_critic_values = []
        self.episode_step_number = 0
        self.actor_loss_values = None #todo debug
        self.critic_loss_values = None #todo debug

    def step(self):
        """Runs a step within a game including a learning step if required"""
        while not self.done:
            self.pick_and_conduct_action_and_save_log_probabilities()
            if self.time_to_learn():
                self.actor_learn()
            self.set_state(self.get_next_state()) #this is to set the state for the next iteration
            self.episode_step_number += 1
        #self.see_updated_probabilities() #todo put if for debug setting
        self.episode_number += 1


    """ Network -> Environment """
    def pick_and_conduct_action_and_save_log_probabilities(self):
        """Picks and then conducts actions. Then saves the log probabilities of the actions it conducted to be used for
        learning later"""
        action, action_log_probability, critic_value = self.pick_action_and_get_log_probabilities_and_critic_value()
        self.set_action(action)
        self.store_action_log_probability(action_log_probability)
        self.store_critic_value(critic_value)
        #print("critic =" + str(critic_value[0]) + " for \n " +  str(self.get_state()))
        self.conduct_action(self.get_action())

    def pick_action_and_get_log_probabilities_and_critic_value(self):
        """Picks actions and then calculates the log probabilities of the actions it picked given the policy"""
        state = torch.from_numpy(self.get_state()).float().unsqueeze(0).to(self.device)
        action_values, critic_value = self.policy(state) #todo questionable cpu gpu
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
        self.optimizer.step()
        
    def time_to_learn(self):
        """Tells us whether it is time for the algorithm to learn. With REINFORCE we only learn at the end of every
        episode so this just returns whether the episode is over"""
        return self.done

    
    """ Calculate Loss """
    def calculate_policy_loss_on_episode(self, total_discounted_reward):
        alpha = 2e-7 
        all_discounted_returns = torch.tensor(self.calculate_discounted_returns())
        all_critic_values = torch.cat(self.get_episode_critic_values(),dim=1)

        advantages = all_discounted_returns - all_critic_values 
        advantages = advantages.detach()

        critic_loss = (all_discounted_returns - all_critic_values)**2
        self.critic_loss_values = critic_loss
        critic_loss = critic_loss.mean() * alpha
        self.critic_loss_debug = critic_loss 

        action_log_probabilities_for_all_episodes = torch.cat(self.episode_log_probabilities)
        actor_loss = -1 * action_log_probabilities_for_all_episodes * advantages
        self.actor_loss_values = actor_loss
        actor_loss =   actor_loss.mean() * alpha
        self.actor_loss_debug = actor_loss
        
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


    """ storage in lists """
    def store_critic_value(self,critic_value):
        self.episode_critic_values.append(critic_value)

    """ get in lists """
    def get_episode_critic_values(self):
        return self.episode_critic_values


    """ aux """
    def see_updated_probabilities(self):
        r = self.get_reward()
        
        for s,a,l,c,aloss,closs in zip(self.get_episode_states(),self.get_episode_actions(),self.get_episode_action_log_probabilities(),self.get_episode_critic_values(),self.actor_loss_values[0],self.critic_loss_values[0]):
            state = torch.from_numpy(s).float().unsqueeze(0).to(self.device)
            actor_values, critic_value = self.policy(state)
            text = """"\r D_reward {0: .2f}, action: {1: 2d}, | old_prob: {2: .5f}, new_prob: {3: .5f} |, >old_crit: {4: .5f}, new_crit: {5: .2f}<, *loss_actor: {6: .3f}, loss_crit: {7: .3f}*"""
            print(text.format(r,a,math.exp(l),actor_values[0][a].item(),c[0].item(),critic_value.item(),aloss.item(),closs.item()))
        print("================================================\n\n")

    
    def see_state(self):
        return self.policy(torch.from_numpy(self.get_state()).float().unsqueeze(0).to(self.device))
    
