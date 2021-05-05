import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from agents.Learning_Agent import Learning_Agent, Config_Learning_Agent


class Config_Reinforce(Config_Learning_Agent):
    def __init__(self,config=None):
        Config_Learning_Agent.__init__(self,config)
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


class REINFORCE(Learning_Agent):
    agent_name = "REINFORCE"
    def __init__(self, config):
        Learning_Agent.__init__(self, config)
        self.policy = self.config.architecture()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    *                            MAIN INTERFACE                               
    *            Main interface to be used by every implemented agent               
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    #TODO implement this
    def play(self,observations:np.array=None,policy=None,info=None) -> tuple([np.array,dict]):
        return NotImplementedError
    
    """
    Methods for Step:
        * step
        * pick action
        * save step info
    """
    def step(self):
        self.action, info = self.pick_action()
        self.action_probability = info["action_probability"]
        self.action_log_probability = info["action_log_probability"]
        self.next_observation, self.reward, self.done, _ = self.environment.step(self.action)
        if self.config.get_clip_rewards(): self.reward =  max(min(self.reward, 1.0), -1.0)
        
    def pick_action(self,current_observation=None) -> tuple([int,dict]):
        if current_observation is None: current_observation = self.observation
        input_state = torch.from_numpy(current_observation).float().unsqueeze(0).to(self.device)
        input_mask = torch.from_numpy(self.mask).unsqueeze(0).to(self.device)
        action_values_logits = self.policy(input_state,input_mask,False)
        action_values_softmax = torch.softmax(action_values_logits,dim=1)
        action_distribution = Categorical(action_values_softmax) # this creates a distribution to sample from
        action = action_distribution.sample()
        return action.item(), {"action_probability": torch.softmax(action_values_softmax,dim=1)[0][action],
            "action_log_probability":torch.log_softmax(action_values_logits,dim=1)[0][action]}

    def save_step_info(self):
        super().save_step_info()
        self.episode_action_probabilities.append(self.action_probability)
        self.episode_action_log_probabilities.append(self.action_log_probability)

    """
    Methods for Learn:
        * learn
        * calculate_policy_loss_on_episode
        * calculate_discounted_returns
    """
    def learn(self):
        raise ValueError(",,")
        policy_loss = self.calculate_policy_loss_on_episode()
        self.take_optimisation_step(self.optimizer,self.policy,policy_loss,self.config.get_gradient_clipping_norm())
        self.log_updated_probabilities()

    def calculate_policy_loss_on_episode(self,episode_action_log_probabilities=None,episode_rewards=None,discount_rate=None):
        raise ValueError(",,")
        if episode_rewards is None: episode_rewards = self.episode_rewards
        if discount_rate is None: discount_rate = self.config.get_discount_rate()
        if episode_action_log_probabilities is None: episode_action_log_probabilities = self.episode_action_log_probabilities

        all_discounted_returns = torch.tensor(self.calculate_discounted_episode_returns(episode_rewards=episode_rewards,discount_rate=discount_rate))

        ''' advantages are just logprob * reward '''
        advantages = all_discounted_returns
        #self.advantages = self.all_discounted_returns - torch.cat(episode_critic_values)

        action_log_probabilities_for_all_episodes = torch.cat(episode_action_log_probabilities)
        actor_loss_values = -1 * action_log_probabilities_for_all_episodes * advantages
        actor_loss =   actor_loss_values.mean()
        return actor_loss

    def calculate_discounted_episode_returns(self,episode_rewards=None,discount_rate=None):
        raise ValueError(",,")
        if episode_rewards is None: episode_rewards = self.episode_rewards
        if discount_rate is None: discount_rate = self.config.get_discount_rate()
        discounted_returns = []
        discounted_total_reward = 0.
        for ix in range(len(episode_rewards)):
            discounted_total_reward = episode_rewards[-(ix + 1)] + discount_rate*discounted_total_reward
            discounted_returns.insert(0,discounted_total_reward)
        return discounted_returns

    



    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    *                            OTHER METHODS                              
    *                          auxiliary methods             
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    ''' clone '''
    def clone(self):
        cloned_agent = REINFORCE(self.config)
        self.copy_model_over(self.policy, cloned_agent.policy)
        return cloned_agent

    """ debug """        
    def log_updated_probabilities(self,print_results=False):
        r = self.reward
        full_text = []
        for s,a,l in zip(self.episode_observations,self.episode_actions,self.episode_action_log_probabilities):
            with torch.no_grad():
                state = torch.from_numpy(s).float().unsqueeze(0).to(self.device)
                mask = torch.tensor(self.environment.environment.get_mask(observation=s))
                prob = torch.softmax(self.policy(state,mask=mask,apply_softmax=False),dim=1)
            text = """\r D_reward {0: .2f}, action: {1: 2d}, | old_prob: {2: .10f}, new_prob: {3: .10f}"""
            formatted_text = text.format(r,a,math.exp(l),prob[0][a].item())
            if(print_results): print(formatted_text)
            full_text.append(formatted_text )
        self.logger.info("Updated probabilities and Loss After update:" + ''.join(full_text))




