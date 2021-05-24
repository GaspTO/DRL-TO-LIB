from torch._C import Value
from environments.Arena import Arena
from agents.Learning_Agent import Learning_Agent, Config_Learning_Agent
from agents.search_agents.Abstract_Network_Search_Agent import Abstract_Network_Search_Agent
from torch.distributions import Categorical
from collections import namedtuple
from time import sleep, time
import torch.optim as optim
import torch
import numpy as np
import random
from torch.utils.data import Dataset



class Config_Tree_Dual_Policy_Iteration(Config_Learning_Agent):
    def __init__(self,config=None):
        super().__init__(config)
        if(isinstance(config,Config_Tree_Dual_Policy_Iteration)):
            self.start_updating_at_episode = config.get_start_updating_at_episode()
            self.update_episode_perodicity = config.get_update_episode_perodicity()
            self.learn_epochs = config.get_learn_epochs()
            self.batches_per_epoch = config.get_batches_per_epoch()
            self.max_transition_memory = config.get_max_transition_memory()
            self.num_data_workers = config.get_num_data_workers()

        else:
            self.start_updating_at_episode = None
            self.update_episode_perodicity = None
            self.learn_epochs = None
            self.batches_per_epoch = None
            self.max_transition_memory = None
            self.num_data_workers = None

    def get_start_updating_at_episode(self):
        if self.start_updating_at_episode is None:
            raise ValueError("start_updating_at_episode can't be None")
        return self.start_updating_at_episode

    def get_update_episode_perodicity(self):
        if self.update_episode_perodicity is None:
            raise ValueError("update_episode_perodicity can't be None")
        return self.update_episode_perodicity

    def get_learn_epochs(self):
        if self.learn_epochs is None:
            raise ValueError("learn_epochs can't be None")
        return self.learn_epochs
        
    def get_batches_per_epoch(self):
        if self.batches_per_epoch is None:
            raise ValueError("batches_per_epoch")
        else:
            return self.batches_per_epoch

    def get_max_transition_memory(self):
        if self.max_transition_memory is None:
            raise ValueError("max_transition_memory can't be None")
        return self.max_transition_memory

    def get_num_data_workers(self):
        if self.num_data_workers is None:
            raise ValueError("num_data_workers can't be None")
        return self.num_data_workers




class Tree_Dual_Policy_Iteration(Learning_Agent):
    agent_name = "Tree_Dual_Policy_Iteration"
    def __init__(self,network, tree_agent, config):
        Learning_Agent.__init__(self, config)

        self.network = network
        self.tree_agent = tree_agent
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #self.network = self.config.architecture(self.device,18,9,1000)
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=2e-05,weight_decay=1e-5)

        self.episodes = []
        self.transitions = []



    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    *                            MAIN INTERFACE                               
    *            Main interface to be used by every implemented agent               
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def step(self):
        self.start = time()
        self.action = self.tree_agent.play(np.array([self.observation]))[0][0]
        self.next_observation, self.reward, self.done, _ = self.environment.step(self.action)


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

    def get_transition_training_data(self,samples) -> Dataset:
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
        return self.done and self.episode_number % self.config.get_update_episode_perodicity() == 0 and self.episode_number >= self.config.get_start_updating_at_episode()

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
    *                         Arena / Network managements             
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""




    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    *                            Logs             
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def log_iteration_text(self,i,debug=False):
        reward_txt = "reward \t{0: .2f} \n".format(self.episode_rewards[i])
        action_txt = "agent_action: \t{1: 2d} \n".format(self.episode_actions[i])
        modified_observation = self.episode_observations[i][0] + -1*self.episode_observations[i][1]
        return reward_txt + action_txt + modified_observation

    def log_final(self,debug=False):
        text = "final-state:\n"
        modified_observation = str(-1*self.episode_next_observations[-1][0] + self.episode_next_observations[-1][1]) + "\n"
        return text + modified_observation

    def log_updated_probabilities(self):
        full_text = []
        for i in range(len(self.episode_observations)):
            iter_text = self.log_iteration_text(i,debug=False)
            full_text.append(iter_text)
        full_text.append(self.log_final(debug=False))
        self.logger.info("Updated probabilities and Loss After update:\n\n" + ''.join(full_text))
    



