import math
import numpy as np
import torch
import torch.nn as nn
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


class Policy_Re(nn.Module):
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

    def forward(self, x, mask=None):
        # in lightning, forward defines the prediction/inference actions
        self.x1 = x.view(x.size(0),-1)
        self.x2 = self.one(self.x1)


        self.actions1 = self.actions(self.x2)
        if(mask is not None):
            #self.actions1 = self.actions1.mul((1-mask)*1e-8 + mask)
            self.actions1 = self.actions1.mul(mask)
        self.actions2 = torch.softmax(self.actions1,dim=1)
        
        #self.actions3 = (self.actions2 + smoothing) 
        #self.actions4 = self.actions3/self.actions3.sum()

 
        self.x2.retain_grad()
        #self.actions1.retain_grad()
        self.actions2.retain_grad()
        #self.actions3.retain_grad()
        #self.actions4.retain_grad()
        
        #critic = self.critic(self.x2)
        #return self.actions2,critic


        return self.actions2

    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters(), lr=2e-13)
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        return optimizer
    

class REINFORCE(Base_Agent):
    agent_name = "REINFORCE"
    def __init__(self, config):
        Base_Agent.__init__(self, config)
        self.policy = Policy_Re()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=2e-5)

    """ Basic Operations """
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
                self.learn()
            self.set_state(self.get_next_state())
            self.episode_step_number += 1
        self.episode_number += 1
        if(self.debug_mode):
            self.log_updated_probabilities()


    """ Network -> Environment """
    def pick_and_conduct_action_and_save_log_probabilities(self):
        """Picks and then conducts actions. Then saves the log probabilities of the actions it conducted to be used for
        learning later"""
        action, log_probabilities = self.pick_action_and_get_log_probabilities()
        self.set_action(action)
        self.store_action_log_probabilities(log_probabilities)
        self.conduct_action(self.get_action())
    

    def pick_action_and_get_log_probabilities(self):
        """Picks actions and then calculates the log probabilities of the actions it picked given the policy"""
        state = torch.from_numpy(self.get_state()).float().unsqueeze(0).to(self.device)
        action_values = self.policy(state)
        action_values_copy = action_values.detach()
        if(self.action_mask_required == True): #todo can't use the forward for this mask cause... critic_output
            mask = self.get_action_mask()
            unormed_action_values_copy =  action_values_copy.mul(mask)
            action_values_copy =  unormed_action_values_copy/unormed_action_values_copy.sum()
        action_distribution = Categorical(action_values_copy) # this creates a distribution to sample from
        action = action_distribution.sample()
        if(self.debug_mode): self.logger.info("Q values\n {} -- Action chosen {} Masked_Prob {:.5f} True_Prob {:.5f}".format(action_values, action.item(),action_values_copy[0][action].item(),action_values[0][action].item()))
        else: self.logger.info("Action chosen {} Masked_Prob {:.5f} True_Prob {:.5f}".format(action.item(),action_values_copy[0][action].item(),action_values[0][action].item()))
        return action.item(), torch.log(action_values[0][action])

    
    """ Learn """
    def learn(self):
        #todo use base_agent: take optimization step
        """Runs a learning iteration for the policy"""
        policy_loss = self.calculate_policy_loss_on_episode()
        self.take_optimisation_step(self.optimizer,self.policy,policy_loss,self.config.get_gradient_clipping_norm())

    def time_to_learn(self):
        """Tells us whether it is time for the algorithm to learn. With REINFORCE we only learn at the end of every
        episode so this just returns whether the episode is over"""
        return self.done


    """ Calculate Loss """
    def calculate_policy_loss_on_episode(self,alpha=1):
        all_discounted_returns = torch.tensor(self.calculate_discounted_returns())

        advantages = all_discounted_returns
        advantages = advantages.detach()

        action_log_probabilities_for_all_episodes = torch.cat(self.get_episode_action_log_probabilities())
        actor_loss_values = -1 * action_log_probabilities_for_all_episodes * advantages
        actor_loss =   actor_loss_values.mean() * alpha
        if(self.debug_mode):
            self.set_debug_variables(actor_loss_values_debug=actor_loss_values,\
                actor_loss_debug=actor_loss)

        return actor_loss

    def calculate_discounted_returns(self):
        discounted_returns = []
        discounted_reward = 0
        for ix in range(len(self.episode_rewards)):
            discounted_reward = self.episode_rewards[-(ix + 1)] + self.config.get_discount_rate()*discounted_reward
            discounted_returns.insert(0,discounted_reward)
        return discounted_returns


    """ storage in lists """
    def store_action_log_probabilities(self, action_log_probabilities):
        """Stores the log probability of picked actions to be used for learning later"""
        self.episode_action_log_probabilities.append(action_log_probabilities)

    """ get in lists """
    def get_episode_action_log_probabilities(self):
        return self.episode_action_log_probabilities


    """ debug """
    def set_debug_variables(self,**args):
        if('actor_loss_values_debug' in args):
            self.actor_loss_values_debug = args['actor_loss_values_debug']
        if('actor_loss_debug' in args):
            self.actor_loss_debug = args['actor_loss_debug']
        
    def log_updated_probabilities(self,print_results=False):
        r = self.get_reward()
        full_text = []
        for s,a,l,aloss in zip(self.get_episode_states(),self.get_episode_actions(),self.get_episode_action_log_probabilities(),self.actor_loss_values_debug):
            with torch.no_grad():
                state = torch.from_numpy(s).float().unsqueeze(0).to(self.device)
            actor_values = self.policy(state)
            text = """\r D_reward {0: .2f}, action: {1: 2d}, | old_prob: {2: .10f}, new_prob: {3: .10f} | *loss_actor: {4: .3f}*"""
            formatted_text = text.format(r,a,math.exp(l),actor_values[0][a].item(),aloss.item())
            if(print_results): print(formatted_text)
            full_text.append(formatted_text )
        self.logger.info("Updated probabilities and Loss After update:" + ''.join(full_text))

    def see_state(self):
        return self.policy(torch.from_numpy(self.get_state()).float().unsqueeze(0).to(self.device))




