
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
from agents.Neural_Agent import *
from agents.networks.v_networks.V_Network import V_Network

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
'''
TREE DUAL POLICY ITERATION
'''
from agents.tree_dual_policy_iteration.Tree_Dual_Policy_Iteration import Config_Tree_Dual_Policy_Iteration, Tree_Dual_Policy_Iteration
from agents.tree_dual_policy_iteration.TDPI_Terminal_Learning import TDPI_Terminal_Learning
'''
SEARCH
'''
from agents.search_agents.K_BFS_Minimax_Agent_Collection import K_Best_First_Minimax_Rollout, K_Best_First_Minimax_V, K_Best_First_Minimax_Q




board_shape = 3
target_length = 3
seed = random.randint(1, 1000)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = V_Network(device,3,3,hidden_nodes=300)
print("seed=" + str(seed))




""" Config """
config = Config()
config.debug_mode = True
config.environment = Custom_Simple_Playground(Custom_K_Row(board_shape=board_shape, target_length=target_length),play_first=True)
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
config_Learning_Agent.architecture =  net
config_Learning_Agent.input_dim = None 
config_Learning_Agent.output_size = None
config_Learning_Agent.is_mask_needed = True
config.random_episodes_to_run = 0
config.epsilon_decay_rate_denominator = 1

""" Config_Tree_Dual_Policy_Iteration """
config_tree_dual_policy_iteration = Config_Tree_Dual_Policy_Iteration(config_Learning_Agent)
config_tree_dual_policy_iteration.start_updating_at_episode = 1
config_tree_dual_policy_iteration.update_episode_perodicity = 100
config_tree_dual_policy_iteration.learn_epochs = 5
config_tree_dual_policy_iteration.max_transition_memory = 20000

#! OUTDATED BELOW
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




#agent = Tree_Dual_Policy_Iteration(config_reinforce)

#tree_agent = K_Best_First_Minimax_Rollout(config.environment.environment,k=2,num_iterations=20,num_rollouts_per_node=2,debug=False)
tree_agent = K_Best_First_Minimax_V(config.environment.environment,k=2,num_iterations=50,network=net)
agent = TDPI_Terminal_Learning(net,tree_agent,config_tree_dual_policy_iteration)

#config_reinforce.environment.add_agent(MCTS_Simple_RL_Agent(config_reinforce.environment.environment,n_iterations=100,network=agent.policy,device=agent.device))
config.environment.add_agent(MCTS_Search(config.environment.environment,n_iterations=100))
game_scores, rolling_scores, time_taken = agent.run_n_episodes(num_episodes=10000)










