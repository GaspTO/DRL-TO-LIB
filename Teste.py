import os
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))

import gym
from environments.Gomoku import GomokuEnv
#from environments.Four_Rooms_Environment import Four_Rooms_Environment

from utilities.data_structures.Config import Config
from agents.Base_Agent import Base_Agent, Config_Base_Agent
from agents.policy_gradient_agents.REINFORCE import REINFORCE, Config_Reinforce
from agents.DQN_agents.DDQN import DDQN, Config_DDQN
from agents.DQN_agents.DQN import DQN, Config_DQN
from agents.actor_critic_agents.A3C import A3C, Config_A3C


""" Config """
config = Config()
config.debug_mode = False
config.environment = GomokuEnv('black','beginner',9)
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
config.seed = 1
config.show_solution_score = False
config.standard_deviation_results = 1.0
config.use_GPU = False


""" Config_Base_Agent """
config_base_agent = Config_Base_Agent(config)
config_base_agent.batch_size = 1
config_base_agent.clip_rewards = False
config_base_agent.architecture = (("Linear",30,"Sigmoid"),("Linear",30,"Sigmoid"),("Linear",30,"Sigmoid"))
config_base_agent.input_dim = None 
config_base_agent.output_size = None
config_base_agent.is_mask_needed = True
config.random_episodes_to_run = 0
config.epsilon_decay_rate_denominator = 1

""" Config_Reinforce """
config_reinforce = Config_Reinforce(config_base_agent)
config_reinforce.discount_rate = 0.99
config_reinforce.learning_rate = 1

""" Config DQN """
config_DQN = Config_DQN(config_base_agent)
config_DQN.buffer_size = 40000
config_DQN.discount_rate = 0.99
config_DQN.gradient_clipping_norm = 0.7
config_DQN.learning_iterations = 1
config_DQN.learning_rate = 0.01
config_DQN.update_every_n_steps = 1



agent = REINFORCE(config_reinforce)
agent = DQN(config_DQN)
game_scores, rolling_scores, time_taken = agent.run_n_episodes(num_episodes=1000)


















'''


config.seed = 1
#config.environment = gym.make("CartPole-v0")
#config.environment = GomokuEnv('black','random',9)
config.environment = GomokuEnv('black','beginner',9)

config.num_episodes_to_run = 1000
config.file_to_save_data_results = 
config.file_to_save_results_graph = 
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 1
config.use_GPU = False
config.overwrite_existing_results_file = False
config.randomise_random_seed = True
config.save_model = False
config.architecture = (("Linear",30,"Sigmoid"),("Linear",30,"Sigmoid"),("Linear",30,"Sigmoid"))








config_Reinforce = Config_Reinforce(config)
config_Reinforce.discount_rate = 0.99
config_Reinforce.learning_rate = 0.01

'''
'''
config.hyperparameters = {
        "architecture":(
            ("Linear",30,"Sigmoid"),("Linear",30,"Sigmoid"),("Linear",30,"Sigmoid")
        ),
        "learning_rate": 0.01,
        "batch_size": 256,
        "buffer_size": 40000,
        "epsilon": 1.0,
        "epsilon_decay_rate_denominator": 1,
        "discount_rate": 0.99,
        "tau": 0.01,
        "alpha_prioritised_replay": 0.6,
        "beta_prioritised_replay": 0.1,
        "incremental_td_error": 1e-8,
        "update_every_n_steps": 1,
        "linear_hidden_units": [30, 15],
        "final_layer_activation": "None",
        "batch_norm": False,
        "gradient_clipping_norm": 0.7,
        "learning_iterations": 1,
        "clip_rewards": False,
        "normalise_rewards": True,
        "exploration_worker_difference": 2.0,
    }
'''
'''

if __name__ == "__main__":
    #agent = DQN_With_Fixed_Q_Targets(config)
    #agent = DDQN(config)
    #agent = DDQN_With_Prioritised_Experience_Replay(config)
    #agent = Dueling_DDQN(config)
    agent = REINFORCE(config)
    #agent = A3C(config)
    game_scores, rolling_scores, time_taken = agent.run_n_episodes(num_episodes=300)





'''

