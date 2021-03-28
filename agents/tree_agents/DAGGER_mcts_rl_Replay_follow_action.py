from numpy.core.numeric import cross
from agents.Base_Agent import Base_Agent, Config_Base_Agent
from agents.policy_gradient_agents.REINFORCE import REINFORCE, Config_Reinforce
from agents.tree_agents.MCTS_Search import MCTS_Agent
from agents.tree_agents.MCTS_RL_Search import MCTS_RL_Agent
from torch.distributions import Categorical
from collections import namedtuple
from time import sleep, time

#from agents.tree_agents.MCTS_RL_Search import MCTS_RL_Agent

#from agents.tree_agents.MCTS_RL_Search import MCTS_RL_Agents
#from agents.tree_agents.Searchfuck import MCTS_Search_attempt_muzero


import torch.optim as optim
import torch
import numpy as np
import math
from utilities.data_structures.Replay_Buffer import Replay_Buffer

Data = namedtuple('Data', ['state', 'action','mask'])

class DAGGER(REINFORCE):
    agent_name = "DAGGER"
    def __init__(self, config, expert = None):
        REINFORCE.__init__(self, config)
        self.policy = self.config.architecture()
        self.expert = expert
        self.optimizer = optim.Adam(self.policy.parameters(), lr=2e-05)
        self.trajectories = []
        self.dataset = []
        self.mask_dataset = []

    def reset_game(self):
        super().reset_game()
        self.episode_action_log_probability_vectors = []
        self.episode_action_expert_suggested_log_prob = []
        self.episode_expert_actions = []

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
        #self.expert_action = self.mcts(self.state) 
        self.expert_action = self.mcts_rl(self.state)
        self.mask = self.get_action_mask()
        self.action, self.log_probabilities, self.expert_log_prob = self.pick_action(expert_action=self.expert_action)
        #self.expert_probabilities = torch.tensor(self.mcts_rl(self.state))
        
        self.next_state, self.reward, done, _ = self.environment.step(self.action)
        if self.config.get_clip_rewards(): self.reward =  max(min(self.reward, 1.0), -1.0)
        self.done = done
        self.save_update_information()

    def pick_action(self,expert_action,current_state=None):
        if current_state is None: current_state = self.state
        input_state = torch.from_numpy(current_state).float().unsqueeze(0).to(self.device)
        action_values_logits = self.policy(input_state,self.get_action_mask())
        action_values_copy =  torch.softmax(action_values_logits,dim=1)
        '''
        if(self.action_mask_required == True): #todo can't use the forward for this mask cause... critic_output
            mask = self.get_action_mask()
            unormed_action_values_copy =  action_values_copy.mul(mask)
            action_values_copy =  unormed_action_values_copy/unormed_action_values_copy.sum()
        '''
        action_distribution = Categorical(action_values_copy) # this creates a distribution to sample from
        action = action_distribution.sample()
        return action.item(), torch.log_softmax(action_values_logits,dim=1)[0][action],torch.log_softmax(action_values_logits,dim=1)[0][torch.tensor([expert_action])]


    def save_update_information(self):
        self.episode_actions.append(self.action)
        self.episode_next_states.append(self.next_state)
        self.episode_rewards.append(self.reward)
        self.episode_dones.append(self.done)
        self.episode_action_log_probability_vectors.append(self.log_probabilities)
        self.episode_action_expert_suggested_log_prob.append(self.expert_log_prob)
        self.episode_expert_actions.append(self.expert_action)
        self.dataset.append(Data(self.state,self.expert_action,self.mask))
        self.total_episode_score_so_far += self.reward

    def mcts_rl(self,state):
        #todo some things in here need config
        search = MCTS_RL_Agent(self.environment.environment,100,self.policy,self.device,exploration_weight=5.0,logger=self.logger)
        action = search.play(state)
        return action

    def mcts(self,state):
        #todo some things in here need config
        search = MCTS_Agent(self.environment.environment,100,exploration_weight=5.0)
        action = search.play(state)
        return action

    def learn(self):
        policy = self.policy
        n = len(self.dataset) + 1
        start = time()
        #todo size of batch for config
         
        size_of_batch = len(self.episode_actions) #todo change
        size_of_batch = 6
        for index in range(size_of_batch,n,size_of_batch):
            states = np.stack([self.dataset[i][0] for i in range(index-size_of_batch,index)])
            actions = np.stack([self.dataset[i][1] for i in range(index-size_of_batch,index)])
            masks = torch.tensor(np.stack([self.dataset[i][2] for i in range(index-size_of_batch,index)]))
            assert len(states) == len(actions) and len(masks) == len(actions) and len(actions) == size_of_batch
            input_data = torch.from_numpy(states).float().to("cuda" if torch.cuda.is_available() else "cpu")
            output = self.policy(input_data,mask=masks)
            output_log = torch.log_softmax(output,dim=1)
            assert type(actions[0]) == np.int64
            loss = [-1 * output_log[idx][actions[idx]] for idx in range(size_of_batch)]
            policy_loss = torch.stack(loss).mean()
            self.take_optimisation_step(self.optimizer,policy,policy_loss,self.config.get_gradient_clipping_norm())
            #todo this 500 is for config
        #self.dataset = self.dataset[-500:]
        
        self.dataset = self.dataset[-20:]
        self.logger.info("time:{0:.10f}".format(time()-start))
        self.log_updated_probabilities()

        
        

    def time_to_learn(self):
        return self.done        
    
    def log_updated_probabilities(self,print_results=False):
        r = self.reward
        full_text = []
        for s,a,ae,elp in zip(self.episode_states,self.episode_actions,self.episode_expert_actions,self.episode_action_expert_suggested_log_prob):
            with torch.no_grad():
                state = torch.from_numpy(s).float().unsqueeze(0).to(self.device)
                mask = torch.tensor(self.environment.environment.get_mask(observation=s))
            prob = torch.softmax(self.policy(state,mask=mask),dim=1)
            after_action_prob = prob[0][ae]
            prob_list = ["{0}=>{1:.2f}".format(i,prob[0][i]) for i in range(len(prob[0]))]
            text = """\r D_reward {0: .2f}, action: {1: 2d}, expert_action: {2: 2d} | expert_prev_prob:{3: .10f} expert_new_prob: {4: .10f} ||| probs: {5}"""
            formatted_text = text.format(r,a,ae,torch.exp(elp).item(),after_action_prob.item(),prob_list)
            if(print_results): print(formatted_text)
            full_text.append(formatted_text )
        self.logger.info("Updated probabilities and Loss After update:" + ''.join(full_text))
    



