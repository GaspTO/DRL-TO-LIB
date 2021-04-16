import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from agents.Learning_Agent import Learning_Agent, Config_Learning_Agent
from agents.STI.Search_Evaluation_Function import UCT, PUCT
from agents.STI.Tree_Policy import Tree_Policy, Greedy_DFS, Adversarial_Greedy_Best_First_Search, Local_Greedy_DFS_With_Global_Restart
from agents.STI.Expansion_Strategy import Expansion_Strategy, One_Successor_Rollout, Network_One_Successor_Rollout
from agents.STI.Astar_minimax import Astar_minimax


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


    
class NEW_REINFORCE(Learning_Agent):
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
    """
    def step(self):
        self.action, info = self.pick_action()
        self.action = self.astar_minimax(observation=self.environment.get_current_observation())
        self.action_probability = info["action_probability"]
        self.action_log_probability = info["action_log_probability"]
        self.next_observation, self.reward, self.done, _ = self.environment.step(self.action)
        if self.config.get_clip_rewards(): self.reward =  max(min(self.reward, 1.0), -1.0)
        
    def pick_action(self,current_observation=None) -> tuple([int,dict]):
        if current_observation is None: current_observation = self.observation
        input_state = torch.from_numpy(current_observation).float().unsqueeze(0).to(self.device)
        mask = self.environment.get_mask()
        action_values = self.policy(input_state,torch.tensor(mask).unsqueeze(0),apply_softmax=False)
        action_values_copy = torch.softmax(action_values,dim=1)
        action_distribution = Categorical(action_values_copy) # this creates a distribution to sample from
        action = action_distribution.sample()
        return action.item(), {"action_probability": torch.softmax(action_values,dim=1)[0][action],
            "action_log_probability":torch.log_softmax(action_values,dim=1)[0][action]}

    """
    Methods for Learn:
        * learn
        * calculate_policy_loss_on_episode
        * calculate_discounted_returns
    """
    def learn(self):
        policy_loss = self.calculate_policy_loss_on_episode(episode_rewards=self.episode_rewards,episode_log_probs=self.episode_action_log_probabilities)
        self.take_optimisation_step(self.optimizer,self.policy,policy_loss,self.config.get_gradient_clipping_norm())
        self.log_updated_probabilities()

    def calculate_policy_loss_on_episode(self,alpha=1,episode_rewards=None,episode_log_probs=None,episode_critic_values=None):
        if episode_rewards is None: episode_rewards = self.episode_rewards
        if episode_log_probs is None: episode_log_probs = self.episode_action_log_probabilities

        self.all_discounted_returns = torch.tensor(self.calculate_discounted_returns(episode_rewards=episode_rewards))
        self.advantages = self.all_discounted_returns - torch.cat(episode_critic_values)

        action_log_probabilities_for_all_episodes = torch.cat(episode_log_probs)
        actor_loss_values = -1 * action_log_probabilities_for_all_episodes * self.advantages
        self.actor_loss =   actor_loss_values.mean() * alpha
        return self.actor_loss

    def calculate_discounted_returns(self,episode_rewards=None):
        if episode_rewards is None: episode_rewards = self.episode_rewards
        discounted_returns = []
        for ix in range(len(episode_rewards)):
            discounted_reward = episode_rewards[-(ix + 1)] + self.config.get_discount_rate()*discounted_reward
            discounted_returns.insert(0,discounted_reward)
        return discounted_returns

    """
    method for save_step_info
    """
    def save_step_info(self):
        super().save_step_info()
        self.episode_action_probabilities.append(self.action_probability)
        self.episode_action_log_probabilities.append(self.action_log_probability)



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
                input_state = torch.from_numpy(s).float().unsqueeze(0).to(self.device)
            actor_values = torch.softmax(self.policy(input_state),dim=1)
            text = """\r D_reward {0: .2f}, action: {1: 2d}, | old_prob: {2: .10f}, new_prob: {3: .10f}"""
            formatted_text = text.format(r,a,math.exp(l),actor_values[0][a].item())
            if(print_results): print(formatted_text)
            full_text.append(formatted_text )
        self.logger.info("Updated probabilities and Loss After update:" + ''.join(full_text))

    def astar_minimax(self,observation):
        env = self.environment.environment
        eval_fn = PUCT()
        agent = Astar_minimax(env,self.policy,self.device,eval_fn)
        act = agent.play(observation=observation)
        return act

