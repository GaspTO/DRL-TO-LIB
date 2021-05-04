from random import shuffle
from numpy.core.numeric import cross
from torch._C import Value
import agents.tree_agents.MCTS_Agents as MCTS_Agents
from agents.Learning_Agent import Learning_Agent, Config_Learning_Agent
from agents.STI.Tree_Search_Iteration import Tree_Search_Iteration
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
from agents.STI.Search_Evaluation_Function import UCT, PUCT
from agents.STI.Tree_Policy import Tree_Policy, Greedy_DFS, Adversarial_Greedy_Best_First_Search, Local_Greedy_DFS_With_Global_Restart
from agents.STI.Expansion_Strategy import *
from agents.STI.Astar_minimax import Astar_minimax


Data = namedtuple('Data', ['observation', 'action','mask'])
Episode_Tuple = namedtuple('Episode_Tuple', ['episode_observations','episode_masks','episode_actions','episode_expert_actions','episode_expert_action_tree_probs','episode_rewards'])

class ALPHAZERO(Learning_Agent):
    agent_name = "ALPHAZERO"
    def __init__(self, config, expert = None):
        Learning_Agent.__init__(self, config)
        self.network = self.config.architecture()
        self.expert = expert
        self.optimizer = optim.Adam(self.network.parameters(), lr=2e-05)
        self.trajectories = []
        self.dataset = []
        self.mask_dataset = []
        self.episode_data = []
        self.memory_size = 20
        self.size_of_batch = 6
        self.debug = False 
        self.episodes = []


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
        self.expert_action, self.expert_action_tree_probs = self.try_expert(self.observation,25,1,1.0)
        ''' '''

        self.action, info = self.pick_action()
        self.action_log_probability = info["action_log_probability"]
        self.expert_action_probability = torch.softmax(info["logits"],dim=1)[0][torch.tensor([self.expert_action])]
        self.expert_action_log_probability = torch.log_softmax(info["logits"],dim=1)[0][torch.tensor([self.expert_action])]
        #! EXPERT
        self.next_observation, self.reward, self.done, _ = self.environment.step(self.expert_action)

    def pick_action(self,current_observation=None) -> tuple([int,dict]):
        if current_observation is None: current_observation = self.observation
        else: raise ValueError("Right now pick action can only use internal state")
        input_state = torch.from_numpy(current_observation).float().unsqueeze(0).to(self.device)
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
            "logits": policy_values_logits}


    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    *                            LEARNING METHODS     
    *                       Learning on Trajectories                                 
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def time_to_learn(self):
        return self.done and self.episode_number % 100 == 0 and self.episode_number != 0

    def learn(self):
        episodes_copy = copy.deepcopy(self.episodes)
        shuffle(episodes_copy)
        for i in range(8):
            for episode in episodes_copy:
                batch_x = torch.FloatTensor(episode.episode_observations)
                self.network.load_state(batch_x)
                loss_policy_on_tree = self.learn_expert_action_probs(episode)
                loss_value_on_trajectory = self.learn_trajectory_reward(episode)
                total_loss = loss_policy_on_tree + loss_value_on_trajectory
                self.take_optimisation_step(self.optimizer,self.network,total_loss,self.config.get_gradient_clipping_norm())
        self.episodes = self.episodes[:500]

    def learn_expert_action_probs(self,episode):
        masks = torch.Tensor(episode.episode_masks)
        network_policy_values_logits = self.network.get_policy_values(apply_softmax=False,mask=masks)
        network_log_policy_values = torch.log_softmax(network_policy_values_logits,dim=1).reshape(-1)
        target_policy_values = torch.FloatTensor(episode.episode_expert_action_tree_probs).reshape(-1)
        loss = -1 * target_policy_values.dot(network_log_policy_values)
        return loss


    def learn_trajectory_reward(self,episode):
        def calculate_discounted_episode_returns(episode_rewards,discount_rate=1):
            discounted_returns = []
            discounted_total_reward = 0.
            for ix in range(len(episode_rewards)):
                discounted_total_reward = episode_rewards[-(ix + 1)] + discount_rate*discounted_total_reward
                discounted_returns.insert(0,discounted_total_reward)
            return discounted_returns

        state_value = self.network.get_state_value()
        discounted_rewards = torch.tensor(calculate_discounted_episode_returns(episode.episode_rewards,discount_rate=1))
        loss_vector = (state_value - discounted_rewards)**2
        loss = loss_vector.mean()
        return loss


    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    *                      EPISODE/STEP DATA MANAGEMENT        
    *                   Manages step, episode and reset data                                   
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def save_step_info(self):
        super().save_step_info()
        self.episode_expert_actions.append(self.expert_action)
        self.episode_expert_action_tree_probs.append(self.expert_action_tree_probs) 
        self.episode_expert_action_probabilities.append(self.expert_action_probability)
        self.episode_expert_action_log_probabilities.append(self.expert_action_log_probability)        
        self.dataset.append(Data(self.observation,self.expert_action,self.mask))

    def end_episode(self):
        super().end_episode()
        self.log_updated_probabilities(print_results=self.debug)
        self.episodes.append(Episode_Tuple
            (self.episode_observations,
            self.episode_masks,
            self.episode_actions,
            self.episode_expert_actions,
            self.episode_expert_action_tree_probs,
            self.episode_rewards))
       # ['episode_observations','episode_masks','episode_actions','episode_expert_actions','episode_expert_action_tree_probs','episode_rewards'])

    def reset(self):
        super().reset()
        self.episode_expert_actions = []
        self.episode_expert_action_tree_probs = []
        self.episode_expert_action_probabilities = []
        self.episode_expert_action_log_probabilities = [] 
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

    def try_expert(self,observation,n_rounds,k,exploration_weight):
        env = self.environment.environment
        #* eval functions
        #eval_fn = UCT()
        eval_fn = PUCT()

        #* tree policy 
        tree_policy =  Greedy_DFS(evaluation_fn=eval_fn)
        #tree_policy = Adversarial_Greedy_Best_First_Search(evaluation_fn=eval_fn)
        #tree_policy = Local_Greedy_DFS_With_Global_Restart(evaluation_fn=eval_fn)
        
        #* expand policy
        #tree_expansion = One_Successor_Rollout()
        tree_expansion = Network_One_Successor_Rollout(self.network,self.device)
        #tree_expansion = Network_Policy_Value(self.network,self.device)
        #tree_expansion = Normal_With_Network_Estimation(self.network,self.device)

        agent = Tree_Search_Iteration(env,playout_iterations=n_rounds,tree_policy=tree_policy,tree_expansion=tree_expansion,search_expansion_iterations=k)
        action_probs = agent.play(observation=observation)
        #action_rl = self.mcts_simple_rl(observation,100,1.0)

        #!sampling
        action_distribution = Categorical(torch.tensor(action_probs)) # this creates a distribution to sample from
        action = action_distribution.sample()
        #action = action_probs.argmax()
        return action, action_probs
                
    
    def astar_minimax(self,observation):
        env = self.environment.environment
        eval_fn = PUCT()
        agent = Astar_minimax(env,self.network,self.device,eval_fn)
        act = agent.play(observation=observation)
        return act
    



    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    *                            Other Methods...             
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def log_updated_probabilities(self,print_results=False):
        r = self.reward
        full_text = []
        for s,a,ae,elp,treep in zip(self.episode_observations,self.episode_actions,self.episode_expert_actions,self.episode_expert_action_log_probabilities,self.episode_expert_action_tree_probs):
            with torch.no_grad():
                state = torch.from_numpy(s).float().unsqueeze(0).to(self.device)
                mask = torch.tensor(self.environment.environment.get_mask(observation=s))
                self.network.load_state(state)
                prob = self.network.get_policy_values(mask=mask,apply_softmax=True)
                value = self.network.get_state_value()
            after_action_prob = prob[0][ae]
            prob_list = ["{0}=>{1:.2f}".format(i,prob[0][i]) for i in range(len(prob[0]))]
            tree_prob_list = ["{0}=>{1:.2f}".format(i,treep[i]) for i in range(len(treep))]
            text = """D_reward {0: .2f}, Value_Estimate {6: .2f} | agent_action: {1: 2d}, expert_action: {2: 2d} | expert_prev_prob:{3: .10f} expert_new_prob: {4: .10f} |||\n net_vec: {5}\n tree_vec: {7}\n"""
            formatted_text = text.format(r,a,ae,torch.exp(elp).item(),after_action_prob.item(),prob_list,value.item(),tree_prob_list)
            formatted_text = formatted_text + str(s) + "\n"
            if(print_results): print(formatted_text)
            full_text.append(formatted_text )
        self.logger.info("Updated probabilities and Loss After update:\n" + ''.join(full_text))
    



