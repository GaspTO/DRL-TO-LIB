from numpy.core.numeric import cross
from torch._C import Value
from agents.Learning_Agent import Learning_Agent, Config_Learning_Agent
import agents.tree_agents.MCTS_Agents as MCTS_Agents
from torch.distributions import Categorical
from collections import namedtuple
from time import sleep, time
import torch.optim as optim
import torch
import numpy as np
import math
from utilities.data_structures.Replay_Buffer import Replay_Buffer


class Config_DAGGER(Config_Learning_Agent):
    def __init__(self,config=None):
        Config_Learning_Agent.__init__(self,config)
        if(isinstance(config,Config_DAGGER)):
            self.learning_rate = config.get_learning_rate()
        else:
            self.learning_rate = 1 

    def get_learning_rate(self):
        if(self.learning_rate == None):
            raise ValueError("Learning Rate Not Defined")
        return self.learning_rate

Data = namedtuple('Data', ['observation', 'action','mask'])

class DAGGER(Learning_Agent):
    agent_name = "DAGGER"
    def __init__(self, config, expert = None):
        Learning_Agent.__init__(self, config)
        self.policy = self.config.architecture()
        self.expert = expert
        self.optimizer = optim.Adam(self.policy.parameters(), lr=2e-05)
        self.trajectories = []
        self.dataset = []
        self.mask_dataset = []
        self.memory_size = 20
        self.size_of_batch = 6


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
        self.expert_action = self.mcts_simple_rl(self.observation,100)
        self.action, info = self.pick_action()
        self.action_log_probability = info["action_log_probability"]
        self.expert_action_probability = torch.softmax(info["logits"],dim=1)[0][torch.tensor([self.expert_action])]
        self.expert_action_log_probability = torch.log_softmax(info["logits"],dim=1)[0][torch.tensor([self.expert_action])]
        self.next_observation, self.reward, self.done, _ = self.environment.step(self.action)

    def pick_action(self,current_observation=None) -> tuple([int,dict]):
        if current_observation is None: current_observation = self.observation
        else: raise ValueError("Right now pick action can only use internal state")
        input_state = torch.from_numpy(current_observation).float().unsqueeze(0).to(self.device)
        input_mask = torch.from_numpy(self.mask).unsqueeze(0).to(self.device)
        action_values_logits = self.policy(input_state,input_mask,False)
        action_values_softmax =  torch.softmax(action_values_logits,dim=1)
        action_distribution = Categorical(action_values_softmax) # this creates a distribution to sample from
        action = action_distribution.sample()
        return action.item(), {"action_probability": action_values_softmax[0][action],
            "action_log_probability":torch.log_softmax(action_values_logits,dim=1)[0][action],
            "logits": action_values_logits}


    """
    Methods to Learn:
        * learn
    """
    def learn(self):
        policy = self.policy
        n = len(self.dataset) + 1
        start = time()
        size_of_batch = self.size_of_batch
        for index in range(size_of_batch,n,size_of_batch):
            observations = np.stack([self.dataset[i][0] for i in range(index-size_of_batch,index)])
            actions = np.stack([self.dataset[i][1] for i in range(index-size_of_batch,index)])
            masks = torch.tensor(np.stack([self.dataset[i][2] for i in range(index-size_of_batch,index)]))
            assert len(observations) == len(actions) and len(masks) == len(actions) and len(actions) == size_of_batch
            input_data = torch.from_numpy(observations).float().to("cuda" if torch.cuda.is_available() else "cpu")
            output = self.policy(input_data,mask=masks,apply_softmax=False)
            output_log = torch.log_softmax(output,dim=1)
            assert type(actions[0]) == np.int64
            loss = [-1 * output_log[idx][actions[idx]] for idx in range(size_of_batch)]
            policy_loss = torch.stack(loss).mean()
            self.take_optimisation_step(self.optimizer,policy,policy_loss,self.config.get_gradient_clipping_norm())
        self.dataset = self.dataset[-self.memory_size:]
        self.logger.info("time:{0:.10f}".format(time()-start))
        

    """
    Override:
        * save_step_info
        * end_episode
        * reset
    """
    def save_step_info(self):
        super().save_step_info()
        self.episode_expert_actions.append(self.expert_action)
        self.episode_expert_action_probabilities.append(self.expert_action_probability)
        self.episode_expert_action_log_probabilities.append(self.expert_action_log_probability)        
        self.dataset.append(Data(self.observation,self.expert_action,self.mask))


    def end_episode(self):
        super().end_episode()
        self.log_updated_probabilities()

    def reset(self):
        super().reset()
        self.episode_expert_actions = []
        self.episode_expert_action_probabilities = []
        self.episode_expert_action_log_probabilities = [] 
        #! CAREFUL
        self.environment.play_first = self.environment.play_first == False
        

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    *                            EXPERT AGENTS                              
    *            Main interface to be used by every implemented agent               
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def mcts(self,observation,n):
        #todo some things in here need config
        search = MCTS_Agents.MCTS_Agent(self.environment.environment,n,exploration_weight=5.0)
        action = search.play(observation)
        return action

    def mcts_simple_rl(self,observation,n):
        #todo some things in here need config
        search = MCTS_Agents.MCTS_Simple_RL_Agent(self.environment.environment,n,self.policy,self.device,exploration_weight=5.0)
        action = search.play(observation)
        return action

    def mcts_exploratory_rl(self,observation,n):
        search = MCTS_Agents.MCTS_Exploratory_RL_Agent(self.environment.environment,n,self.policy,self.device,exploration_weight=5.0)
        action = search.play(observation)
        return action



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
            text = """\r D_reward {0: .2f}, action: {1: 2d}, expert_action: {2: 2d} | expert_prev_prob:{3: .10f} expert_new_prob: {4: .10f} ||| probs: {5}\n"""
            formatted_text = text.format(r,a,ae,torch.exp(elp).item(),after_action_prob.item(),prob_list)
            if(print_results): print(formatted_text)
            full_text.append(formatted_text )
        self.logger.info("Updated probabilities and Loss After update:" + ''.join(full_text))
    



