from agents.policy_gradient_agents.REINFORCE import Config_Reinforce, REINFORCE
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions import Categorical
from agents.Base_Agent import Base_Agent, Config_Base_Agent
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

    def forward(self, x):
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
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        return optimizer
    

class REINFORCE_Baseline(REINFORCE):
    agent_name = "REINFORCE_Baseline"
    def __init__(self, config):
        REINFORCE.__init__(self, config)
        #self.policy = self.create_NN_through_NNbuilder(input_dim=self.input_shape, output_size=self.action_size + 1,smoothing=0.001)
        self.policy = Policy_ReBa()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=2e-4)
        #self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-10)
        #self.optimizer = optim.SGD(self.policy.parameters(), lr=1e-3)
        

    """ Basic Operations """
    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        super().reset_game()
        self.episode_critic_values = []

    def do_episode(self):
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
        action, action_log_probabilities, critic_value = self.pick_action_and_get_log_probabilities_and_critic_value()
        self.set_action(action)
        self.store_action_log_probabilities(action_log_probabilities)
        self.store_critic_value(critic_value)
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
        if(self.debug_mode): self.logger.info("Q values\n {} -- Action chosen {} Masked_Prob {:.5f} True_Prob {:.5f} Critic_value {:.5f}".format(action_values, action.item(),action_values_copy[0][action].item(),action_values[0][action].item(),critic_value.item()))
        else: self.logger.info("Action chosen {} Masked_Prob {:.5f} True_Prob {:.5f} Critic_value {:.5f}".format(action.item(),action_values_copy[0][action].item(),action_values[0][action].item(),critic_value.item()))
        return action.item(), torch.log(action_values[0][action]), critic_value


    """ Calculate Loss """
    def calculate_policy_loss_on_episode(self,alpha=1):
        all_discounted_returns = torch.tensor(self.calculate_discounted_returns())
        all_critic_values = torch.cat(self.get_episode_critic_values(),dim=1).view(-1)

        advantages = all_discounted_returns - all_critic_values 
        advantages = advantages.detach()

        critic_loss_values = (all_discounted_returns - all_critic_values)**2
        critic_loss = critic_loss_values.mean() * alpha 

        action_log_probabilities_for_all_episodes = torch.cat(self.get_episode_action_log_probabilities())
        actor_loss_values = -1 * action_log_probabilities_for_all_episodes * advantages
        actor_loss =   actor_loss_values.mean() * alpha

        total_loss = actor_loss + critic_loss

        if(self.debug_mode):
            self.set_debug_variables(critic_loss_values_debug=critic_loss_values, \
                critic_loss_debug=critic_loss, actor_loss_values_debug=actor_loss_values,\
                actor_loss_debug=actor_loss)
            self.logger.info("Actor_Loss: {} and Critic_Loss: {}".format(actor_loss,critic_loss))
        return total_loss


    """ storage in lists """
    def store_critic_value(self,critic_value):
        self.episode_critic_values.append(critic_value)

    """ get in lists """
    def get_episode_critic_values(self):
        return self.episode_critic_values


    """ debug """
    def set_debug_variables(self,**args):
        if('critic_loss_values_debug' in args):
            self.critic_loss_values_debug = args['critic_loss_values_debug']
        if('critic_loss_debug'  in args):
            self.critic_loss_debug = args['critic_loss_debug']
        if('actor_loss_values_debug' in args):
            self.actor_loss_values_debug = args['actor_loss_values_debug']
        if('actor_loss_debug' in args):
            self.actor_loss_debug = args['actor_loss_debug']
        

    def log_updated_probabilities(self,print_results=False):
        full_text = []
        for r,s,a,l,c,aloss,closs in zip(self.get_episode_rewards(),self.get_episode_states(),self.get_episode_actions(),self.get_episode_action_log_probabilities(),self.get_episode_critic_values(),self.actor_loss_values_debug,self.critic_loss_values_debug):
            state = torch.from_numpy(s).float().unsqueeze(0).to(self.device)
            actor_values, critic_value = self.policy(state)
            text = """"\r D_reward {0: .2f}, action: {1: 2d}, | old_prob: {2: .10f}, new_prob: {3: .10f} |, >old_crit: {4: .5f}, new_crit: {5: .5f}<, *loss_actor: {6: .3f}, loss_crit: {7: .3f}*"""
            formatted_text = text.format(r,a,math.exp(l),actor_values[0][a].item(),c[0].item(),critic_value.item(),aloss.item(),closs.item())
            if(print_results): print(formatted_text)
            full_text.append(formatted_text )
        self.logger.info("Updated probabilities and Loss After update:" + ''.join(full_text))


    def see_state(self):
        return self.policy(torch.from_numpy(self.get_state()).float().unsqueeze(0).to(self.device))
    
