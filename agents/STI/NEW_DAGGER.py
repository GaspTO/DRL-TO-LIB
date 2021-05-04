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
from utilities.data_structures.Replay_Buffer import Replay_Buffer
''' '''
from agents.STI.Search_Evaluation_Function import UCT, PUCT
from agents.STI.Tree_Policy import Tree_Policy, Greedy_DFS, Adversarial_Greedy_Best_First_Search, Local_Greedy_DFS_With_Global_Restart
from agents.STI.Expansion_Strategy import Expansion_Strategy, One_Successor_Rollout, Network_One_Successor_Rollout
from agents.STI.Astar_minimax import Astar_minimax


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

class NEW_DAGGER(Learning_Agent):
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
        self.debug = False 

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
        #self.expert_action = self.try_expert(self.observation,25,5,1.0)
        self.expert_action = self.mcts_simple_rl(self.observation,100,1.0)
        self.stri_action = self.mcts_original(self.observation,100,1.0)
        #self.expert_action = self.astar_minimax(self.observation)
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
        action_values_logits = self.policy.load_state(input_state).get_policy_values(False,input_mask)
        action_values_softmax =  torch.softmax(action_values_logits,dim=1)
        #! what is the purpose of sampling in DAGGER?
        #action_distribution = Categorical(action_values_softmax) # this creates a distribution to sample from
        #action = action_distribution.sample()
        action = action_values_softmax.argmax()
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
            output = self.policy.load_state(input_data).get_policy_values(apply_softmax=False,mask=masks)
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
        self.log_updated_probabilities(print_results=self.debug)

    def reset(self):
        super().reset()
        #! DEBUG
        self.episode_stri = []
        self.episode_stri_mcts_normal= []

        self.episode_expert_actions = []
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
        self.episode_stri_mcts_normal.append(search.stri)
        return action
    
    def mcts_simple_rl(self,observation,n,exploration_weight):
        #todo some things in here need config
        search = MCTS_Agents.MCTS_Simple_RL_Agent(self.environment.environment,n,self.policy,self.device,exploration_weight=exploration_weight)
        action = search.play(observation)
        self.episode_stri.append(search.stri)
        return action
    '''
    def mcts_exploitation_rl(self,observation,n,exploration_weight):
        search = MCTS_Agents.MCTS_Exploitation_RL_Agent(self.environment.environment,n,self.policy,self.device,exploration_weight=exploration_weight)
        action = search.play(observation)
        return action

    def mcts_IDAstar_rl(self,observation,n,exploration_weight):
        search = MCTS_Agents.MCTS_IDAstar_Agent(self.environment.environment,n,self.policy,self.device,exploration_weight=exploration_weight)
        action = search.play(observation)
        return action
    '''

    def try_expert(self,observation,n_rounds,k,exploration_weight):
        env = self.environment.environment
        #* eval functions
        #eval_fn = UCT()
        eval_fn = PUCT()

        #* tree policy 
        #tree_policy =  Greedy_DFS(evaluation_fn=eval_fn)
        tree_policy = Adversarial_Greedy_Best_First_Search(evaluation_fn=eval_fn)
        #tree_policy = Local_Greedy_DFS_With_Global_Restart(evaluation_fn=eval_fn)
        
        #* expand policy
        #tree_expansion = One_Successor_Rollout()
        tree_expansion = Network_One_Successor_Rollout(self.policy,self.device)

        agent = Tree_Search_Iteration(env,playout_iterations=n_rounds,tree_policy=tree_policy,tree_expansion=tree_expansion,search_expansion_iterations=k)
        action = agent.play(observation=observation)
        #action_rl = self.mcts_simple_rl(observation,100,1.0)
        raise ValueError("don't take max action but sample")
        return action
                
    
    def astar_minimax(self,observation):
        env = self.environment.environment
        eval_fn = PUCT()
        agent = Astar_minimax(env,self.policy,self.device,eval_fn)
        act = agent.play(observation=observation)
        return act
    



    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    *                            Other Methods...             
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def log_updated_probabilities(self,print_results=False):
        r = self.reward
        full_text = []
        for s,a,ae,elp,st,st_n in zip(self.episode_observations,self.episode_actions,self.episode_expert_actions,self.episode_expert_action_log_probabilities,self.episode_stri,self.episode_stri_mcts_normal):
            with torch.no_grad():
                state = torch.from_numpy(s).float().unsqueeze(0).to(self.device)
                mask = torch.tensor(self.environment.environment.get_mask(observation=s))
            prob = self.policy.load_state(state).get_policy_values(mask=mask,apply_softmax=True)
            after_action_prob = prob[0][ae]
            prob_list = ["{0}=>{1:.2f}".format(i,prob[0][i]) for i in range(len(prob[0]))]
            text = """D_reward {0: .2f}, agent_action: {1: 2d}, expert_action: {2: 2d} | expert_prev_prob:{3: .10f} expert_new_prob: {4: .10f} ||| probs: {5}\n"""
            text = text + st + "\n"
            text = text + "++++++++++++++++++++++++++++++++++++++++++++\n" + st_n + "\n"
            formatted_text = text.format(r,a,ae,torch.exp(elp).item(),after_action_prob.item(),prob_list)
            if(print_results): print(formatted_text)
            full_text.append(formatted_text )
        self.logger.info("Updated probabilities and Loss After update:\n" + ''.join(full_text))
    



