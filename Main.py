
from agents.tree_agents.MCTS_Search import MCTS_Search
import os
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
import gym
import torch
import torch.nn as nn
import random

'''
NEURAL NETWORK
'''
from agents.Neural_Agent import Policy_Value_MLP, Double_Policy_Value_MLP

'''
Environments
'''
from environments.gomoku.Gomoku import GomokuEnv
from environments.Custom_K_Row import Custom_K_Row
#from environments.Custom_Cart_Pole import Custom_Cart_Pole
from environments.core.Custom_Simple_Playground import Custom_Simple_Playground
from environments.core.Simple_Self_Play import Simple_Self_Play
'''
BASES
'''
from utilities.data_structures.Config import Config
from agents.Learning_Agent import Learning_Agent, Config_Learning_Agent
'''
DQN
'''
from agents.DQN_agents.DQN import DQN, Config_DQN
from agents.DQN_agents.DDQN import DDQN, Config_DDQN
from agents.DQN_agents.DQN_With_Fixed_Q_Targets import Config_DQN_With_Fixed_Q_Targets
'''
POLICY BASED
'''
from agents.policy_gradient_agents.REINFORCE import REINFORCE, Config_Reinforce
from agents.policy_gradient_agents.REINFORCE_BASELINE import REINFORCE_BASELINE, Config_Reinforce_Baseline
from agents.actor_critic_agents.A3C import A3C, Config_A3C


'''
TREE BASED
'''
from agents.tree_agents.MCTS_Agents import MCTS_Search, MCTS_Simple_RL_Agent
from agents.tree_agents.DAGGER import DAGGER
'''
STI
'''
from agents.STI.ALPHAZERO import ALPHAZERO
'''
ASTAR
'''
#from agents.Simple_Astar.ASTAR_DAGGER import ASTAR_DAGGER



seed = random.randint(1, 1000)
print("seed=" + str(seed))




""" Config """
config = Config()
config.debug_mode = True
config.environment = Custom_K_Row(board_shape=3, target_length=3)
#config.environment = Simple_Playground_Env(K_Row_Interface(board_shape=3, target_length=3))
#config.environment = Simple_Self_Play(episodes_to_update=100,environment=config.enviroig.environment
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
config.use_GPU = False


""" Config_Learning_Agent """
config_Learning_Agent = Config_Learning_Agent(config)
config_Learning_Agent.batch_size = 16
config_Learning_Agent.gradient_clipping_norm = 0.7
config_Learning_Agent.clip_rewards = False
config_Learning_Agent.architecture =  Double_Policy_Value_MLP
config_Learning_Agent.input_dim = None 
config_Learning_Agent.output_size = None
config_Learning_Agent.is_mask_needed = True
config.random_episodes_to_run = 0
config.epsilon_decay_rate_denominator = 1

""" Config_Reinforce """
config_reinforce = Config_Reinforce(config_Learning_Agent)
config_reinforce.discount_rate = 0.99
config_reinforce.learning_rate = 2e-12 #2e-12

""" Config_Reinforce_Baseline """
config_reinforce_baseline = Config_Reinforce_Baseline(config_reinforce)
#config_reinforce_baseline.critic_architecture = Critic
config_reinforce_baseline.critic_learning_rate = 2e-05

""" Config DQN """
config_DQN = Config_DQN(config_Learning_Agent)
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
config_A3C = Config_A3C(config_Learning_Agent)
config_A3C.discount_rate = 0.95
config_A3C.learning_rate = 2e-05
config.exploration_worker_difference = 2.0


""" AGENTS """
config_reinforce.environment = Custom_Simple_Playground(config.environment,play_first=True)

#agent = REINFORCE(config_reinforce)
#agent = REINFORCE_BASELINE(config_reinforce_baseline)
#agent = REINFORCEadv_krow(config_reinforce)
#agent = REINFORCE_Baseline(config_reinforce_baseline)
#agent = Logic_Loss_Reinforce(config_reinforce) 
#agent = REINFORCE_Tree(config_reinforce)
#agent = REINFORCE_Tree_2(config_reinforce)
#agent = REINFORCEadv_krow_mcts_vs_mcts(config_reinforce)
#agent = REINFORCE_adv(config_reinforce)
#agent = REINFORCE_adv_negative(config_reinforce)

#agent = DQN(config_DQN)
#agent = DDQN(config_DDQN)
#agent = DDQN_krow(config_DDQN)
#agent = A3C(config_A3C) 



#agent = DAGGER(config_reinforce)
   
agent = ALPHAZERO(config_reinforce)
#agent = NEW_DAGGER_REINFORCE(config_reinforce) #! <---
#agent = ASTAR_DAGGER(config_reinforce)
torch.autograd.set_detect_anomaly(True)
#config_reinforce.environment.add_agent(MCTS_Simple_RL_Agent(config_reinforce.environment.environment,n_iterations=100,network=agent.policy,device=agent.device))
config_reinforce.environment.add_agent(MCTS_Search(config_reinforce.environment.environment,n_iterations=25))
game_scores, rolling_scores, time_taken = agent.run_n_episodes(num_episodes=100000)
#todo these algorithms don't put new tensors on gpu if asked
#todo need to creat configs
#todo actions should be tensors and not integers
#todo should not use the word state, but observation and next_observation
#todo logger should be global
#todo refactor networks: softmax and mask should be explicitly passed or not passed, to avoid mistakes like forgetting we're suppose to use it
#todo have a more formal way of managing networks
#todo DAGGER like alpazero should keep the tree for the next play













