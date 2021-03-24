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


    
class REINFORCE(Base_Agent):
    agent_name = "REINFORCE"
    def __init__(self, config):
        Base_Agent.__init__(self, config)
        self.policy = self.config.architecture()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)

    """ test time """
    def play(self,states:np.array=None)->np.array:
        if states is None: raise ValueError('Needs a state batch #fixme')
        #print("play on Reinforce is broken - quickfix")
        a,p = self.pick_action_and_get_probability(current_state=states)
        return a

    """ Basic Operations """
    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        super().reset_game()
        self.episode_states.append(self.state)
        self.episode_action_log_probabilities = []
        self.episode_critic_values = []
        self.episode_step_number = 0

    def step(self):
        while not self.done:
            self.conduct_action()
            if self.time_to_learn():
                self.set_terminal_reward() #useful for actor-critic
                self.learn()
            self.episode_step_number += 1
            self.state = self.next_state
            self.episode_states.append(self.state)
        self.episode_number += 1
        if(self.debug_mode):
            self.log_updated_probabilities()
        if(self.done == True):
            self.logger.info("final_reward: {}".format(self.reward))
    
    def conduct_action(self):
        """Conducts an action in the environment"""
        self.action, self.probability = self.pick_action_and_get_probability()
        self.next_state, self.reward, done, _ = self.environment.step(self.action)
        if self.config.get_clip_rewards(): self.reward =  max(min(self.reward, 1.0), -1.0)
        self.done = done
        self.save_update_information()
        
    def save_update_information(self):
        self.critic_value = self.get_critic_value() #needs to be here cause it might need next_state
        self.episode_actions.append(self.action)
        self.episode_next_states.append(self.next_state)
        self.episode_rewards.append(self.reward)
        self.episode_dones.append(self.done)
        self.episode_action_log_probabilities.append(torch.log(self.probability))
        self.episode_critic_values.append(self.critic_value)
        self.total_episode_score_so_far += self.reward

    
    def pick_action_and_get_probability(self,current_state=None):
        if current_state is None: current_state = self.state
        input_state = torch.from_numpy(current_state).float().unsqueeze(0).to(self.device)
        action_values = self.policy(input_state)
        action_values_copy = action_values.detach()
        if(self.action_mask_required == True): #todo can't use the forward for this mask cause... critic_output
            mask = self.get_action_mask()
            unormed_action_values_copy =  action_values_copy.mul(mask)
            action_values_copy =  unormed_action_values_copy/unormed_action_values_copy.sum()
        action_distribution = Categorical(action_values_copy) # this creates a distribution to sample from
        action = action_distribution.sample()
        if(self.debug_mode): self.logger.info("Q values\n {} -- Action chosen {} Masked_Prob {:.5f} True_Prob {:.5f}".format(action_values, action.item(),action_values_copy[0][action].item(),action_values[0][action].item()))
        else: self.logger.info("Action chosen {} Masked_Prob {:.5f} True_Prob {:.5f}".format(action.item(),action_values_copy[0][action].item(),action_values[0][action].item()))
        return action.item(), action_values[0][action]

    def get_critic_value(self):
        return torch.tensor([0.])

    def set_terminal_reward(self):
        ''' useful for actor critics '''
        self.terminal_reward = 0


    """ Learn """
    def learn(self):
        """Runs a learning iteration for the policy"""
        policy = self.policy
        episode_rewards = self.episode_rewards
        episode_log_probs = self.episode_action_log_probabilities
        policy_loss = self.calculate_policy_loss_on_episode(episode_rewards=episode_rewards,episode_log_probs=episode_log_probs)
        self.take_optimisation_step(self.optimizer,policy,policy_loss,self.config.get_gradient_clipping_norm())
        self.log_updated_probabilities()

    def time_to_learn(self):
        """Tells us whether it is time for the algorithm to learn. With REINFORCE we only learn at the end of every
        episode so this just returns whether the episode is over"""
        return self.done


    """ Calculate Loss """
    def calculate_policy_loss_on_episode(self,alpha=1,episode_rewards=None,episode_log_probs=None,episode_critic_values=None):
        if episode_rewards is None: episode_rewards = self.episode_rewards
        if episode_log_probs is None: episode_log_probs = self.episode_action_log_probabilities
        if episode_critic_values is None: episode_critic_values = self.episode_critic_values

        self.all_discounted_returns = torch.tensor(self.calculate_discounted_returns(episode_rewards=episode_rewards))
        self.advantages = self.all_discounted_returns - torch.cat(episode_critic_values)

        action_log_probabilities_for_all_episodes = torch.cat(episode_log_probs)
        actor_loss_values = -1 * action_log_probabilities_for_all_episodes * self.advantages
        self.actor_loss =   actor_loss_values.mean() * alpha
        return self.actor_loss

    def calculate_discounted_returns(self,episode_rewards=None):
        if episode_rewards is None: episode_rewards = self.episode_rewards
        discounted_returns = []
        discounted_reward = self.terminal_reward
        for ix in range(len(episode_rewards)):
            discounted_reward = episode_rewards[-(ix + 1)] + self.config.get_discount_rate()*discounted_reward
            discounted_returns.insert(0,discounted_reward)
        return discounted_returns

    ''' clone '''
    def clone(self):
        cloned_agent = REINFORCE(self.config)
        self.copy_model_over(self.policy, cloned_agent.policy)
        return cloned_agent


    """ debug """        
    def log_updated_probabilities(self,print_results=False):
        r = self.reward
        full_text = []
        for s,a,l in zip(self.episode_states,self.episode_actions,self.episode_action_log_probabilities):
            with torch.no_grad():
                state = torch.from_numpy(s).float().unsqueeze(0).to(self.device)
            actor_values = self.policy(state)
            text = """\r D_reward {0: .2f}, action: {1: 2d}, | old_prob: {2: .10f}, new_prob: {3: .10f}"""
            formatted_text = text.format(r,a,math.exp(l),actor_values[0][a].item())
            if(print_results): print(formatted_text)
            full_text.append(formatted_text )
        self.logger.info("Updated probabilities and Loss After update:" + ''.join(full_text))

    def see_state(self):
        return self.policy(torch.from_numpy(self.state).float().unsqueeze(0).to(self.device))




