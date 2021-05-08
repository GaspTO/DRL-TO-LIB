import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from agents.policy_gradient_agents.REINFORCE import REINFORCE, Config_Reinforce
import agents.tree_agents.MCTS_Agents as MCTS_Agents
from agents.STI.Search_Evaluation_Function import UCT, PUCT
from agents.STI.Tree_Policy import Tree_Policy, Greedy_DFS, Adversarial_Greedy_Best_First_Search, Local_Greedy_DFS_With_Global_Restart
from agents.STI.Expansion_Strategy import Expansion_Strategy, One_Successor_Rollout, Network_One_Successor_Rollout
from agents.STI.Astar_minimax import Astar_minimax





class NEW_DAGGER_REINFORCE(REINFORCE):
    gent_name = "NEW DAGGER REINFORCE"
    def __init__(self, config, expert = None):
        REINFORCE.__init__(self, config)
        self.policy = self.config.architecture()
        self.expert = expert
        self.optimizer = optim.Adam(self.policy.parameters(), lr=2e-05)
        #self.optimizer = optim.SGD(self.policy.parameters(), lr=2e-03)
        self.all_episodes_expert_action_log_probabilities =  []
        self.all_episodes_rewards = []
        self.all_episodes_masks = []


    """
    Methods to Step:
        * step
        * pick action
    """
    def step(self):
        self.expert_action = self.mcts_simple_rl(self.observation,100,1.0)
        self.action, agent_info = self.pick_action()
        #! CAREFUL: using expert action + mcts exploitation
        #self.action = self.expert_action
        '''store probs of experts play and of agent play '''
        self.action_values_softmax = torch.softmax(agent_info["logits"],dim=1)
        self.action_values_log_softmax = torch.log_softmax(agent_info["logits"],dim=1)
        

        self.expert_action_probability = self.action_values_softmax[0][torch.tensor([self.expert_action])]
        self.expert_action_log_probability = self.action_values_log_softmax[0][torch.tensor([self.expert_action])]
        self.action_probability = self.action_values_softmax[0][torch.tensor([self.action])]
        self.action_log_probability = self.action_values_log_softmax[0][torch.tensor([self.action])]
        '''play'''
        self.next_observation, self.reward, self.done, _ = self.environment.step(self.expert_action)

    def pick_action(self,current_observation=None) -> tuple([int,dict]):
        if current_observation is None: current_observation = self.observation
        else: raise ValueError("Right now pick action can only use internal state")
        input_state = torch.from_numpy(current_observation).float().unsqueeze(0).to(self.device)
        input_mask = torch.from_numpy(self.mask).unsqueeze(0).to(self.device)
        action_values_logits = self.policy(input_state,input_mask,False)
        action_values_softmax =  torch.softmax(action_values_logits,dim=1)
        #! what is the purpose of sampling in DAGGER?
        #action_distribution = Categorical(action_values_softmax) # this creates a distribution to sample from
        #action = action_distribution.sample()
        action = action_values_softmax.argmax()
        return action.item(), {"action_probability": action_values_softmax[0][action],
            "action_log_probability":torch.log_softmax(action_values_logits,dim=1)[0][action],
            "logits": action_values_logits}


    def save_step_info(self):
        super().save_step_info()
        self.episode_expert_actions.append(self.expert_action)
        self.episode_expert_action_probabilities.append(self.expert_action_probability)
        self.episode_expert_action_log_probabilities.append(self.expert_action_log_probability)        
        #self.dataset.append(Data(self.observation,self.expert_action,self.mask))

    def mcts_original(self,observation,n,exploration_weight):
        #todo some things in here need config
        search = MCTS_Agents.MCTS_Search(self.environment.environment,n,exploration_weight=exploration_weight)
        action = search.play(observation)
        return action

    def mcts_simple_rl(self,observation,n,exploration_weight):
        #todo some things in here need config
        search = MCTS_Agents.MCTS_Simple_RL_Agent(self.environment.environment,n,self.policy,self.device,exploration_weight=exploration_weight)
        action = search.play(observation)
        return action

    '''
    def learn(self):
        for ep_action_log_probs, ep_rewards in zip(self.all_episodes_expert_action_log_probabilities,self.all_episodes_rewards):
            policy_loss = self.calculate_policy_loss_on_episode(ep_action_log_probs,ep_rewards)
            self.take_optimisation_step(self.optimizer,self.policy,policy_loss,self.config.get_gradient_clipping_norm(), retain_graph=True)
        self.log_updated_probabilities()
    '''

    
    def before_learn_block(self):
        assert  self.done == True
        super().end_episode()
        self.all_episodes_expert_action_log_probabilities.append(self.episode_expert_action_log_probabilities)
        self.all_episodes_rewards.append(self.episode_rewards)
        self.all_episodes_masks.append(self.episode_masks)
        
        self.all_episodes_expert_action_log_probabilities =  self.all_episodes_expert_action_log_probabilities[-7:]
        self.all_episodes_rewards = self.all_episodes_rewards[-7:]
        self.all_episodes_masks = self.all_episodes_masks[-7:]
    

    
    def reset(self):
        super().reset()
        self.episode_expert_actions = []
        self.episode_expert_action_probabilities = []
        self.episode_expert_action_log_probabilities = [] 
        #! CAREFUL
        self.environment.play_first = self.environment.play_first == False

    """""""""""""""""""""""""""""""""""
    Methods for Learn:
        * learn
        * calculate_policy_loss_on_episode
        * calculate_discounted_returns
    """""""""""""""""""""""""""""""""""
    def learn(self):
        policy_loss = self.calculate_policy_loss_on_episode(episode_action_log_probabilities=self.episode_expert_action_log_probabilities,episode_rewards=self.episode_rewards,discount_rate=1.0)
        #! careful
        if policy_loss < 0.: policy_loss = -1 * policy_loss
        self.take_optimisation_step(self.optimizer,self.policy,policy_loss,self.config.get_gradient_clipping_norm())
        self.log_updated_probabilities()

    def calculate_policy_loss_on_episode(self,episode_action_log_probabilities=None,episode_rewards=None,discount_rate=None):
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
        if episode_rewards is None: episode_rewards = self.episode_rewards
        if discount_rate is None: discount_rate = self.config.get_discount_rate()
        discounted_returns = []
        discounted_total_reward = 0.
        for ix in range(len(episode_rewards)):
            discounted_total_reward = episode_rewards[-(ix + 1)] + discount_rate*discounted_total_reward
            discounted_returns.insert(0,discounted_total_reward)
        return discounted_returns
    
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    *                            Other Methods...             
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def log_updated_probabilities(self,print_results=False):
        r = self.reward
        full_text = []
        for s,a,ae,elp in zip(self.episode_observations,self.episode_actions,self.episode_expert_actions,self.episode_expert_action_log_probabilities):
            with torch.no_grad():
                state = torch.from_numpy(s).float().unsqueeze(0).to(self.device)
                mask = torch.tensor(self.environment.environment.get_mask(observation=s))
            prob = torch.softmax(self.policy(state,mask=mask,apply_softmax=False),dim=1)
            after_action_prob = prob[0][ae]
            prob_list = ["{0}=>{1:.2f}".format(i,prob[0][i]) for i in range(len(prob[0]))]
            text = """D_reward {0: .2f}, action: {1: 2d}, expert_action: {2: 2d} | expert_prev_prob:{3: .10f} expert_new_prob: {4: .10f} ||| probs: {5}\n"""
            formatted_text = text.format(r,a,ae,torch.exp(elp).item(),after_action_prob.item(),prob_list)
            if(print_results): print(formatted_text)
            full_text.append(formatted_text )
        self.logger.info("Updated probabilities and Loss After update:\n" + ''.join(full_text))