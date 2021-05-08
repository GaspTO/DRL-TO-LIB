from random import shuffle
import agents.tree_agents.MCTS_Agents as MCTS_Agents
from agents.Learning_Agent import Learning_Agent, Config_Learning_Agent
from torch.distributions import Categorical
from collections import namedtuple
from time import sleep, time
import torch.optim as optim
import torch
import numpy as np
import math
import copy
from utilities.data_structures.Replay_Buffer import Replay_Buffer
''' '''
from agents.STI.Score_Strategy import *
from agents.STI.Evaluation_Strategy import *
from agents.STI.Tree_Policy import *
from agents.STI.Expansion_Strategy import *
from agents.STI.Backup_Strategy import *
from agents.STI.Evaluation_Strategy import *
from agents.STI.Search_Node import *





Episode_Tuple = namedtuple('Episode_Tuple', ['episode_observations','episode_masks','episode_actions','episode_expert_actions','episode_net_actions','episode_expert_action_probability_vector','episode_rewards','episode_expert_state_values'])

class ALPHAZERO(Learning_Agent):
    agent_name = "ALPHAZERO"
    def __init__(self, config, expert = None):
        Learning_Agent.__init__(self, config)
        #!hidden nodes
        self.network = self.config.architecture(18,9,1000).to(torch.device("cpu"))
        self.expert = expert
        #!OPTIMIZE
        self.optimizer1 = optim.Adam(self.network.parameters(), lr=2e-05)
        self.optimizer2 = optim.Adam(self.network.parameters(), lr=2e-06)
        self.optimizerSGD = optim.SGD(self.network.parameters(),lr=2e-07)
        self.trajectories = []
        self.dataset = []
        self.mask_dataset = []
        self.episode_data = []
        self.episodes = []
        self.debug = False 

        self.memory_size = 500 #1
        self.size_of_batch = 6
        self.update_on_episode = 100000 #1
        self.learn_epochs = 8 #1


    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    *                            MAIN INTERFACE                               
    *            Main interface to be used by every implemented agent               
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    #TODO implement this
    def play(self,observations:np.array=None,policy=None,info=None) -> tuple([np.array,dict]):
        return NotImplementedError

    """
    Methods to Step:
        * step
        * pick action
    """
    def step(self):
        self.start = time()
        ''' Agents '''
        #self.expert_action = self.try_expert(self.observation,25,5,1.0)
        #self.expert_action,self.expert_action_tree_probs = self.mcts_simple_rl(self.observation,25,1.0)
        #self.stri_action = self.mcts_original(self.observation,100,1.0)
        #self.expert_action = self.astar_minimax(self.observation)
        #! 
        self.expert_action, self.expert_action_probability_vector, self.expert_state_value = self.try_expert(self.observation,100,1,1.0)
        #self.expert_action, self.expert_action_probability_vector, self.expert_state_value = self.try_expert(self.observation,100,1,1.0)
        ''' '''

        self.net_action, info = self.pick_action()
        self.net_action_probability_vector = info['probability_vector']
        self.net_state_value = info['state_value']
        #self.action_log_probability = info["action_log_probability"]
        #self.expert_action_probability = torch.softmax(info["logits"],dim=1)[0][torch.tensor([self.expert_action])]
        #self.expert_action_log_probability = torch.log_softmax(info["logits"],dim=1)[0][torch.tensor([self.expert_action])]
        #! EXPERT
        self.action = self.expert_action
        self.next_observation, self.reward, self.done, _ = self.environment.step(self.action)

    def pick_action(self,current_observation=None) -> tuple([int,dict]):
        if current_observation is None: current_observation = self.observation
        else: raise ValueError("Right now pick action can only use internal state")
        input_state = torch.from_numpy(current_observation).float().unsqueeze(0).to(torch.device("cpu"))
        input_mask = torch.from_numpy(self.mask).unsqueeze(0).to(self.device)
        self.network.load_state(input_state)
        ''' policy '''
        policy_values_logits = self.network.get_policy_values(False,input_mask)
        policy_values_softmax =  torch.softmax(policy_values_logits,dim=1)
        #!sampling
        #action_distribution = Categorical(policy_values_softmax) # this creates a distribution to sample from
        #action = action_distribution.sample()
        action = policy_values_softmax.argmax()
        ''' state value '''
        state_value = self.network.get_state_value()

        
        return action.item(), {"action_probability": policy_values_softmax[0][action],
            "action_log_probability":torch.log_softmax(policy_values_logits,dim=1)[0][action],
            "logits": policy_values_logits[0], "probability_vector": policy_values_softmax[0], "state_value":state_value[0]}


    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    *                            LEARNING METHODS     
    *                       Learning on Trajectories                                 
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def time_to_learn(self):
        return self.done and self.episode_number % self.update_on_episode == 0 and self.episode_number != 0

    def learn(self):
        episodes_copy = copy.deepcopy(self.episodes)
        shuffle(episodes_copy)
        for i in range(self.learn_epochs):
            for episode in episodes_copy:
                self.network.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                batch_x = torch.FloatTensor(episode.episode_observations).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                self.network.load_state(batch_x)
                loss_policy_on_tree = self.learn_policy_on_tree(episode)
                #loss_value_on_tree = self.learn_value_on_tree(episode)
                #loss_policy_on_trajectory = self.learn_policy_on_trajectory_reinforce(episode)
                loss_value_on_trajectory = self.learn_value_on_trajectory(episode)
                #! ADD TOTAL LOSS
                total_loss = loss_policy_on_tree + loss_value_on_trajectory
                self.take_optimisation_step(self.optimizer1,self.network,total_loss, self.config.get_gradient_clipping_norm())

               
        self.network.to(torch.device("cpu"))
        self.episodes = self.episodes[:self.memory_size]

    def learn_value_on_tree(self,episode):
        network_state_values = self.network.get_state_value()
        target_state_values = torch.cat(episode.episode_expert_state_values)
        loss_vector = (network_state_values - target_state_values)**2
        loss = loss_vector.mean()
        return loss

    def learn_policy_on_tree(self,episode):
        masks = torch.Tensor(episode.episode_masks)
        network_policy_values_logits = self.network.get_policy_values(apply_softmax=False,mask=masks)
        network_log_policy_values = torch.log_softmax(network_policy_values_logits,dim=1).reshape(-1)
        target_policy_values =torch.cat(episode.episode_expert_action_probability_vector)
        loss = -1 * target_policy_values.dot(network_log_policy_values)
        return loss

    def learn_value_on_trajectory(self,episode,discount_rate=1):
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


    def learn_policy_on_trajectory_reinforce(self,episode,discount_rate=1):
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
        masks = torch.Tensor(episode.episode_masks)
        network_policy_values_logits = self.network.get_policy_values(apply_softmax=False,mask=masks)
        network_log_policy_values = torch.log_softmax(network_policy_values_logits,dim=1)
        actions = torch.tensor(episode.episode_actions)
        log_action_values =  network_log_policy_values[torch.arange(len(network_log_policy_values)),actions]
        loss_vector = -1 * log_action_values
        loss = loss_vector.mean()
        return loss


    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    *                      EPISODE/STEP DATA MANAGEMENT        
    *                   Manages step, episode and reset data                                   
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def save_step_info(self):
        super().save_step_info()
        #* actions
        #normal actions are in super
        self.episode_expert_actions.append(self.expert_action)
        self.episode_net_actions.append(self.net_action)
        #* vectors
        self.episode_expert_action_probability_vector.append(self.expert_action_probability_vector)
        self.episode_net_action_probability_vector.append(self.net_action_probability_vector)
        #* action probability 
        #self.episode_expert_action_probabilities.append(self.expert_action_probability)
        #self.episode_expert_action_log_probabilities.append(self.expert_action_log_probability)
        #* state value
        self.episode_net_state_values.append(self.net_state_value)
        self.episode_expert_state_values.append(self.expert_state_value)
        #* save trajectories
        if self.done == True:
             self.episodes.append(Episode_Tuple
                (self.episode_observations,
                self.episode_masks,
                self.episode_actions,
                self.episode_expert_actions,
                self.episode_net_actions,
                self.episode_expert_action_probability_vector,
                self.episode_rewards,
                self.episode_expert_state_values))

    def end_episode(self):
        super().end_episode()
        self.log_updated_probabilities(print_results=self.debug)
       
       # ['episode_observations','episode_masks','episode_actions','episode_expert_actions','episode_expert_action_tree_probs','episode_rewards'])

    def reset(self):
        super().reset()
        self.episode_expert_actions = []
        self.episode_net_actions = []
        self.episode_expert_action_tree_probs = []
        self.episode_expert_action_probabilities = []
        self.episode_expert_action_log_probabilities = [] 
        self.episode_net_action_probability_vector = []
        self.episode_expert_action_probability_vector = []
        self.episode_net_state_values = []
        self.episode_expert_state_values = []
        #! CAREFUL
        self.environment.play_first = self.environment.play_first == False
        

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    *                            EXPERT AGENTS                              
    *            Main interface to be used by every implemented agent               
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
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
    '''
    def mcts_exploitation_rl(self,observation,n,exploration_weight):
        search = MCTS_Agents.MCTS_Exploitation_RL_Agent(self.environment.environment,n,self.network,self.device,exploration_weight=exploration_weight)
        action = search.play(observation)
        return action

    def mcts_IDAstar_rl(self,observation,n,exploration_weight):
        search = MCTS_Agents.MCTS_IDAstar_Agent(self.environment.environment,n,self.network,self.device,exploration_weight=exploration_weight)
        action = search.play(observation)
        return action
    '''

    def try_expert(self,observation,iterations,k,exploration_weight):
        env = self.environment.environment

        #* score functions
        tree_score = Visit_Count_Score(temperature=1)
        #tree_score = Win_Ratio_Score()

        #* eval functions
        tree_evaluation = UCT(exploration_weight=1.0)
        #tree_evaluation = PUCT(exploration_weight=2.0)

        #* expand policy
        #! expand policy
        tree_expansion = One_Successor_Rollout()
        #tree_expansion = Network_One_Successor_Rollout(self.network,self.device)
        #tree_expansion = Network_Policy_Value(self.network,self.device)
        #tree_expansion = Normal_With_Network_Estimation(self.network,self.device)

        #* backup
        tree_backup = Backup_W_N_one_successor()

        #* tree policy 
        tree_policy = Greedy_DFS_Recursive(self.environment.environment,iterations,
                                        score_st=tree_score,
                                        evaluation_st=tree_evaluation,
                                        expansion_st=tree_expansion,
                                        backup_st=tree_backup)

        #!
        #random.setstate((3,tuple(range(625)),None))
        action_probs, info = tree_policy.play(observation)
        root_node = info["root_node"]

        #!sampling
        #action_distribution = Categorical(torch.tensor(action_probs)) # this creates a distribution to sample from
        #action = action_distribution.sample()
        action = action_probs.argmax()

        return action, torch.FloatTensor(action_probs), torch.tensor([root_node.W/root_node.N])
        

    



    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    *                            Other Methods...             
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def log_updated_probabilities(self,print_results=False):
        r = self.reward
        full_text = []
        for observation,mask,action,expert_action,net_action,reward,expert_state_value,net_state_value,expert_vector,net_vector in zip(
                self.episode_observations,
                self.episode_masks,
                self.episode_actions,
                self.episode_expert_actions,
                self.episode_net_actions,
                self.episode_rewards,
                self.episode_expert_state_values,
                self.episode_net_state_values,
                self.episode_expert_action_probability_vector,
                self.episode_net_action_probability_vector,
                ):

            with torch.no_grad():
                state = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
                self.network.load_state(state)
                new_net_vector = self.network.get_policy_values(apply_softmax=True,mask=torch.tensor(mask).unsqueeze(0))[0]
                new_state_value = self.network.get_state_value()[0]

            expert_vector_string = ["{0}=>{1:.4f}".format(i,expert_vector[i]) for i in range(len(expert_vector))]
            net_vector_string = ["{0}=>{1:.4f}".format(i,net_vector[i]) for i in range(len(net_vector))]
            new_net_vector_string = ["{0}=>{1:.4f}".format(i,new_net_vector[i]) for i in range(len(new_net_vector))]
            modified_observation = observation[0] + -1*observation[1]
            

            text = "reward \t{0: .2f} \n \
            agent_action: \t{1: 2d} \n \
            expert_action: \t{2: 2d} \n \
            net_action: \t{3: 2d} \n \
            taken_action_prev_prob:\t{4: .10f} \n \
            taken_action_after_prob: \t{5: .10f} \n \
            Expert_Value_Estimate: \t {6: .2f} \n \
            before_Net_Value_Estimate: \t{7: .2f} \n \
            after_Net_Value_Estimate: \t{8: .2f} \n \
            expert_____vector: \t{9}\n \
            before_net_vector: \t{10}\n \
            after__net_vector:  \t{11}\n{12}\n--------------------------\n"
            

            formatted_text = text.format(
                reward,
                action, expert_action.item(), net_action,
                net_vector[action].item(), new_net_vector[action].item(),
                expert_state_value.item(),net_state_value.item(), new_state_value.item(),
                expert_vector_string,
                net_vector_string,
                new_net_vector_string,
                modified_observation,
                )
            if(print_results): print(formatted_text)
            full_text.append(formatted_text)
        self.logger.info("Updated probabilities and Loss After update:\n" + ''.join(full_text))
    



