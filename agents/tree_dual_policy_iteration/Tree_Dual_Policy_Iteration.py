import agents.tree_agents.MCTS_Agents as MCTS_Agents
from agents.Learning_Agent import Learning_Agent, Config_Learning_Agent
from torch.distributions import Categorical
from collections import namedtuple
from time import sleep, time
import torch.optim as optim
import torch
import numpy as np
import copy
import random

''' 
MCTS
'''
from agents.search_agents.mcts.MCTS import MCTS
from agents.search_agents.mcts.MCTS_Evaluation_Strategy import *
from agents.search_agents.mcts.MCTS_Expansion_Strategy import *
'''
MINIMAX
'''
from agents.search_agents.minimax.Minimax import Minimax
from agents.search_agents.minimax.Minimax_Value_Estimation_Strategy import *
'''
BEST-FIRST MINIMAX
'''
from agents.search_agents.k_best_first_minimax.K_Best_First_Minimax import K_Best_First_Minimax
from agents.search_agents.k_best_first_minimax.K_Best_First_Minimax_Expansion_Strategy import *

class Config_Tree_Dual_Policy_Iteration(Config_Learning_Agent):
    def __init__(self,config=None):
        super().__init__(config)
        if(isinstance(config,Config_Tree_Dual_Policy_Iteration)):
            self.update_on_episode = config.get_update_on_episode()
            self.learn_epochs = config.get_learn_epochs()
            self.max_episode_memory = config.get_max_episode_memory()
            self.num_episodes_to_sample = config.get_num_episodes_to_sample()
            self.max_transition_memory = config.get_max_transition_memory()
            self.num_transitions_to_sample = config.get_num_transitions_to_sample()
        else:
            self.update_on_episode = 100
            self.learn_epochs = 5 
            self.max_episode_memory = 500 #1
            self.num_episodes_to_sample = 100
            self.max_transition_memory = 1500
            self.num_transitions_to_sample = 300

    def get_update_on_episode(self):
        return self.update_on_episode

    def get_learn_epochs(self):
        return self.learn_epochs
        
    def get_max_episode_memory(self):
        return self.max_episode_memory
    
    def get_num_episodes_to_sample(self):
        return self.num_episodes_to_sample

    def get_max_transition_memory(self):
        return self.max_transition_memory

    def get_num_transitions_to_sample(self):
        return self.num_transitions_to_sample
    
    

class Tree_Dual_Policy_Iteration(Learning_Agent):
    agent_name = "Tree_Dual_Policy_Iteration"
    def __init__(self, config):
        Learning_Agent.__init__(self, config)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #self.network = self.config.architecture(self.device,18,9,1000)
        self.network = self.config.architecture(self.device,3,3,128)
        self.optimizer = optim.Adam(self.network.parameters(), lr=2e-05,weight_decay=1e-5)

        self.update_on_episode = 100
        self.learn_epochs = 5 
        self.batch_size = 1

        self.episodes = []
        self.max_episode_memory = 500 #1
        self.num_episodes_to_sample = 100
        
        self.transitions = []
        self.max_transition_memory = 1500
        self.num_transitions_to_sample = 300


    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    *                            MAIN INTERFACE                               
    *            Main interface to be used by every implemented agent               
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def play(self,observations:np.array=None,policy=None,info=None) -> tuple([np.array,dict]):
        return NotImplementedError

    def step(self):
        return NotImplementedError

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    *                      EPISODE/STEP DATA MANAGEMENT        
    *                   Manages step, episode and reset data                                   
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def end_episode(self):
        super().end_episode()
        self.log_updated_probabilities()
    
    def reset(self):
        super().reset()
        #! CAREFUL
        self.environment.play_first = self.environment.play_first == False
    
    ''' Memory '''
    def get_episode_batch(self,episodes,shuffle_transitions=False):
        training_data = random.choices(self.episodes, k=episodes)
        training_data = [transition for ep in training_data for transition in ep] #flat episodes
        if shuffle_transitions:
            random.shuffle(training_data)
        return training_data
        
    def add_episode_to_memory(self,episode, max_size=0):
        self.episodes.append(episode)
        self.episodes = self.episodes[-max_size:]

    def get_transition_batch(self,samples):
        training_data = random.choices(self.transitions, k=samples)
        return training_data
        
    def add_transition_to_memory(self, transition, max_size=0):
        self.transitions.append(transition)
        self.transitions = self.transitions[-max_size:]


    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    *                            LEARNING METHODS     
    *                       Learning on Trajectories                                 
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    ''' standard learn '''
    def time_to_learn(self):
        #* learn at a certain episode_number when done
        return self.done and self.episode_number % self.update_on_episode == 0 and self.episode_number != 0

    def learn(self):
        raise NotImplementedError

    ''' V(S) '''
    def learn_value_on_tree(self,episode):
        self.network.load_observations(np.array(episode.episode_observations))
        network_state_values = self.network.get_state_value()
        target_state_values = torch.cat(episode.episode_expert_state_values)
        loss_vector = (network_state_values - target_state_values)**2
        loss = loss_vector.mean()
        return loss

    def learn_value_on_trajectory(self,episode,discount_rate=1):
        self.network.load_observations(np.array(episode.episode_observations))
        def calculate_discounted_episode_returns(episode_rewards,discount_rate):
            discounted_returns = []
            discounted_total_reward = 0.
            for ix in range(len(episode_rewards)):
                discounted_total_reward = episode_rewards[-(ix + 1)] + discount_rate*discounted_total_reward
                discounted_returns.insert(0,discounted_total_reward)
            return discounted_returns

        state_values = self.network.get_state_value()
        discounted_rewards = torch.tensor(calculate_discounted_episode_returns(episode.episode_rewards,discount_rate=discount_rate))
        loss_vector = (state_values - discounted_rewards)**2
        loss = loss_vector.mean()
        return loss

    ''' Q(S,A) '''
    def learn_q_on_trajectory_monte_carlo(self,episode,discount_rate=1):
        """ this is montecarlo learning, not temporal difference """
        self.network.load_observations(np.array(episode.episode_observations))
        def calculate_discounted_episode_returns(episode_rewards,discount_rate):
            discounted_returns = []
            discounted_total_reward = 0.
            for ix in range(len(episode_rewards)):
                discounted_total_reward = episode_rewards[-(ix + 1)] + discount_rate*discounted_total_reward
                discounted_returns.insert(0,discounted_total_reward)
            return discounted_returns

        discounted_rewards = torch.tensor(calculate_discounted_episode_returns(episode.episode_rewards,discount_rate=discount_rate))
        actions = torch.tensor(episode.episode_actions)
        q_values = self.network.get_q_values()
        q_values =  q_values[torch.arange(len(q_values)),actions]
        loss_vector = (q_values - discounted_rewards)**2
        loss = loss_vector.mean()
        return loss

    def learn_q_on_temporal_difference(self,episode,discount_rate=1):
        '!careful: it reloads the network'
        self.network.load_observations(np.array(episode.episode_observations))
        actions = torch.tensor(episode.episode_actions)
        q_values = self.network.get_q_values()
        q_values =  q_values[torch.arange(len(q_values)),actions]
        self.network.load_observations(np.array(episode.episode_next_observations))
        with torch.no_grad():
            q_values_next = self.network.get_q_values().max(1)[0]
        q_targets = torch.tensor(episode.episode_rewards) + (discount_rate * q_values_next * torch.tensor(np.where(episode.episode_dones,0,1)))
        loss = (q_values - q_targets)**2
        loss = loss.mean()
        return loss

    ''' P(S,A) '''
    def learn_policy_on_tree(self,episode):
        self.network.load_observations(np.array(episode.episode_observations))
        masks = torch.Tensor(episode.episode_masks)
        network_policy_values_logits = self.network.get_policy_values(apply_softmax=False,mask=masks)
        network_log_policy_values = torch.log_softmax(network_policy_values_logits,dim=1).reshape(-1)
        target_policy_values =torch.cat(episode.episode_expert_action_probability_vector)
        loss = -1 * target_policy_values.dot(network_log_policy_values)
        return loss

    def learn_policy_on_trajectory_reinforce(self,episode,discount_rate=1):
        self.network.load_observations(np.array(episode.episode_observations))
        def calculate_discounted_episode_returns(episode_rewards,discount_rate):
            discounted_returns = []
            discounted_total_reward = 0.
            for ix in range(len(episode_rewards)):
                discounted_total_reward = episode_rewards[-(ix + 1)] + discount_rate*discounted_total_reward
                discounted_returns.insert(0,discounted_total_reward)
            return discounted_returns

        discounted_rewards = torch.tensor(calculate_discounted_episode_returns(episode.episode_rewards,discount_rate=discount_rate))
        masks = torch.Tensor(episode.episode_masks)
        network_policy_values_logits = self.network.get_policy_values(apply_softmax=False,mask=masks)
        network_log_policy_values = torch.log_softmax(network_policy_values_logits,dim=1)
        actions = torch.tensor(episode.episode_actions)
        log_action_values =  network_log_policy_values[torch.arange(len(network_log_policy_values)),actions]
        loss_vector = -1 * log_action_values * discounted_rewards
        loss = loss_vector.mean()
        return loss

    def learn_policy_on_trajectory_dagger(self,episode,discount_rate=1):
        self.network.load_observations(np.array(episode.episode_observations))
        masks = torch.Tensor(episode.episode_masks)
        network_policy_values_logits = self.network.get_policy_values(apply_softmax=False,mask=masks)
        network_log_policy_values = torch.log_softmax(network_policy_values_logits,dim=1)
        actions = torch.tensor(episode.episode_actions)
        log_action_values =  network_log_policy_values[torch.arange(len(network_log_policy_values)),actions]
        loss_vector = -1 * log_action_values
        loss = loss_vector.mean()
        return loss

    
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    *                            AGENTS                              
    *            Main interface to be used by every implemented agent               
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    ''' Network '''
    def network_policy_actions(self,current_observation,mask):
        self.network.load_observations(np.expand_dims(current_observation, axis=0))
        policy_values_logits = self.network.get_policy_values(False,np.array([mask]))
        policy_values_softmax =  torch.softmax(policy_values_logits,dim=1)
        action = policy_values_softmax.argmax()
        return action.item(), {"action_probability": policy_values_softmax[0][action],
            "action_log_probability":torch.log_softmax(policy_values_logits,dim=1)[0][action],
            "logits": policy_values_logits[0], "probability_vector": policy_values_softmax[0]}

    def network_state_value(self,current_observation):
        self.network.load_observations(np.expand_dims(current_observation, axis=0))
        state_value = self.network.get_state_value()
        return None, {"state_value":state_value[0]}

    def network_q_values(self,current_observation):
        self.network.load_observations(np.expand_dims(current_observation, axis=0))
        q_values = self.network.get_q_values()
        action = q_values.argmax()
        return action, {"q_values":q_values[0]}
        
    ''' Original MCTS '''
    def mcts_original(self,observation,n,exploration_weight):
        #todo some things in here need config
        search = MCTS_Agents.MCTS_Search(self.environment.environment,n,exploration_weight=exploration_weight)
        action = search.play(observation)
        return action
    
    def mcts_simple_rl(self,observation,n,exploration_weight):
        #todo some things in here need config
        search = MCTS_Agents.MCTS_Simple_RL_Agent(self.environment.environment,n,self.network,self.device,exploration_weight=exploration_weight)
        action = search.play(observation)
        probs = search.probs
        return action,probs

    ''' New Search agents '''
    def mcts_expert(self,observation,iterations,k,exploration_weight):
        env = self.environment.environment

        #* eval functions
        tree_evaluation = UCT(exploration_weight=1.0)
        #tree_evaluation = UCT_P(exploration_weight=1.0)
        #tree_evaluation = PUCT(exploration_weight=1.0)

        #* expand policy
        #! expand policy
        #tree_expansion = MCTS_One_Successor_Rollout()
        tree_expansion = MCTS_Network_Value(self.network,self.device)
        #tree_expansion = MCTS_Network_Policy_Value(self.network,self.device)
        #tree_expansion = MCTS_Network_Policy_One_Successor_Rollout(self.network,self.device)

 
        #* tree policy 
        tree_policy = MCTS(env,iterations,evaluation_st=tree_evaluation,expansion_st=tree_expansion)
        
        #!
        action_probs, info = tree_policy.play(observation)
        root_node = info["root_node"]

        action = action_probs.argmax()

        return action, torch.FloatTensor(action_probs), torch.tensor([root_node.total_value/root_node.num_visits])
        
    def minimax_expert(self,observation,max_depth):
        env = self.environment.environment

        #* value estimation policy
        #value_estimation = Random_Rollout_Estimation(num_rollouts=1)
        #value_estimation = Network_Value_Estimation(self.network,self.device)
        value_estimation = Network_Q_Estimation(self.network,self.device)

        #* tree policy
        tree_policy = Minimax(env,value_estimation,max_depth=max_depth)
       

        action_probs, info = tree_policy.play(observation)
        root_node = info["root_node"]

        #!sampling
        #action_distribution = Categorical(torch.tensor(action_probs)) # this creates a distribution to sample from
        #action = action_distribution.sample()
        action = action_probs.argmax()

        return action, torch.FloatTensor(action_probs), torch.tensor([root_node.value])
    

    def k_best_first_minimax_expert(self,observation,k,iterations):
        env = self.environment.environment

        #* value estimation policy
        #expansion_st = K_Best_First_All_Successors_Rollout(num_rollouts=1)
        #expansion_st = K_Best_First_Network_Successor_Q(self.network,self.device)
        expansion_st = K_Best_First_Network_Successor_V(self.network,self.device)

        #* tree policy
        tree_policy = K_Best_First_Minimax(env,expansion_st,k=k,num_iterations=iterations)

        #!sampling
        action_probs, info = tree_policy.play(observation)
        root_node = info["root_node"]
        action = action_probs.argmax()

        return action, {"state_value":torch.tensor([root_node.value])}


    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    *                            Other Methods...             
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def log_iteration_text(self,i,debug=False):
        reward_txt = "reward \t{0: .2f} \n".format(self.episode_rewards[i])
        action_txt = "agent_action: \t{1: 2d} \n".format(self.episode_actions[i])
        modified_observation = self.episode_observations[i][0] + -1*self.episode_observation[i][1]
        return reward_txt + action_txt + modified_observation


    def log_updated_probabilities(self):
        full_text = []
        for i in range(len(self.observation)):
            iter_text = self.log_iteration_text(i,debug=False)
            full_text.append(iter_text)

        self.logger.info("Updated probabilities and Loss After update:\n" + ''.join(full_text))
    



