from agents.DQN_agents.DQN_With_Fixed_Q_Targets import Config_DQN_With_Fixed_Q_Targets
import os
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
import gym
import torch
import torch.nn as nn
from environments.Gomoku import GomokuEnv
from environments.K_Row import K_RowEnv
from utilities.data_structures.Config import Config
from agents.Base_Agent import Base_Agent, Config_Base_Agent
from agents.policy_gradient_agents.REINFORCE import REINFORCE, Config_Reinforce
from agents.policy_gradient_agents.REINFORCE_Baseline import REINFORCE_Baseline, Config_Reinforce_Baseline

from logic.Logic_Loss_Reinforce import Logic_Loss_Reinforce
from logic.REINFORCE_Tree import REINFORCE_Tree,  REINFORCE_Tree_2, Config_Reinforce_Tree
from agents.DQN_agents.DDQN import DDQN, Config_DDQN

from agents.DQN_agents.DQN import DQN, Config_DQN
from agents.actor_critic_agents.A3C import A3C, Config_A3C
import random

from boom.REINFORCEadv_krow import REINFORCEadv_krow
from boom.DDQN_krow import DDQN_krow, Config_DDQN_krow
from boom.REINFORCEadv_krow_mcts_vs_mcts import REINFORCEadv_krow_mcts_vs_mcts


seed = random.randint(1, 1000)
print("seed=" + str(seed))

class Policy_Re2(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(50,80),
            nn.ReLU(),
            nn.Linear(80,80),
            nn.ReLU(),
            nn.Linear(80,80),
            nn.ReLU(),
            nn.Linear(80,25)

        )
        '''
        
        self.net = nn.Sequential(
            nn.Conv2d(2,25,2,1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(225,200),
            nn.ReLU(),
            nn.Linear(200,80),
            nn.ReLU(),
            nn.Linear(80,30),
            nn.ReLU(),
            nn.Linear(30,16),
        )
    
        '''
        self.net = nn.Sequential(
            nn.Linear(5*5, 120),
            nn.Sigmoid(),
            nn.Linear(120,120),
            nn.Sigmoid(),
            nn.Linear(120,120),
            nn.Sigmoid(),
            nn.Linear(120,5*5))
        '''

    def forward(self, x, mask=None):
        #self.x1 = x.view(x.size(0),-1)
        self.x1 = x
        self.logits = self.net(self.x1)
        if(mask is not None):
            self.logits = self.logits.mul(mask)
        self.output = torch.softmax(self.logits,dim=1)
        return self.output


""" Config """
config = Config()
config.debug_mode = False
#config.environment = GomokuEnv('black','random',9)
config.environment = K_RowEnv(board_shape=4, target_length=3)
config.file_to_save_data_results = "results/data_and_graphs/Cart_Pole_Results_Data.pkl"
config.file_to_save_results_graph = "results/data_and_graphs/Cart_Pole_Results_Graph.png"
config.hyperparameters = None
config.num_episodes_to_run = 1000
config.overwrite_existing_results_file = False
config.randomise_random_seed = True
config.requirements_to_solve_game = None
config.runs_per_agent = 1
config.visualise_individual_results = False
config.visualise_overall_results = False
config.save_model = False
config.seed = seed
config.show_solution_score = False
config.standard_deviation_results = 1.0
config.use_GPU = True


""" Config_Base_Agent """
config_base_agent = Config_Base_Agent(config)
config_base_agent.batch_size = 16
config_base_agent.gradient_clipping_norm = 0.7
config_base_agent.clip_rewards = False
#config_base_agent.architecture = (("Linear",30,"Sigmoid"),("Linear",30,"Sigmoid"),("Linear",1,"Sigmoid"))
config_base_agent.architecture =  Policy_Re2
config_base_agent.input_dim = None 
config_base_agent.output_size = None
config_base_agent.is_mask_needed = True
config.random_episodes_to_run = 0
config.epsilon_decay_rate_denominator = 1

""" Config_Reinforce """
config_reinforce = Config_Reinforce(config_base_agent)
config_reinforce.discount_rate = 0.95
config_reinforce.learning_rate = 2e-05 

""" Config_Reinforce_Tree """
config_reinforce_tree = Config_Reinforce_Tree(config_reinforce)


""" Config_Reinforce_Baseline """
config_reinforce_baseline = Config_Reinforce_Baseline(config_reinforce)

""" Config DQN """
config_DQN = Config_DQN(config_base_agent)
config_DQN.buffer_size = 40000
config_DQN.discount_rate = 0.99
config_DQN.learning_iterations = 1
config_DQN.learning_rate = 2e-05
config_DQN.update_every_n_steps = 1

""" Config DQN With Fixed Targets """
config_DQN_wft = Config_DQN_With_Fixed_Q_Targets(config_DQN)
config_DQN_wft.tau = 0.01
config_DQN.reset_every_n_steps = 1

""" Config DDQN """
config_DDQN = Config_DDQN(config_DQN_wft)

""" Config A3C """
config_A3C = Config_A3C(config_base_agent)



''' MAIN '''

#agent = REINFORCE(config_reinforce)
#agent = REINFORCEadv_krow(config_reinforce)
#agent = REINFORCE_Baseline(config_reinforce_baseline)
#agent = Logic_Loss_Reinforce(config_reinforce) 
#agent = REINFORCE_Tree(config_reinforce)
#agent = REINFORCE_Tree_2(config_reinforce)
agent = REINFORCEadv_krow_mcts_vs_mcts(config_reinforce)

#agent = DQN(config_DQN)
#agent = DDQN(config_DDQN)
#agent = DDQN_krow(config_DDQN)
#agent = A3C(config_A3C) 

#torch.autograd.set_detect_anomaly(True)
game_scores, rolling_scores, time_taken = agent.run_n_episodes(num_episodes=50000)











