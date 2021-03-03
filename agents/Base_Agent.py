import logging
import os
import sys
import gym
import random
import numpy as np
import torch
import time
# import tensorflow as tf
from nn_builder.pytorch.NN import NN
from utilities.NNbuilder import NNbuilder
from tensorboardX import SummaryWriter
from torch.optim import optimizer

from utilities.data_structures.Config import Config



class Config_Base_Agent(Config):
    def __init__(self,config=None):
        Config.__init__(self,config)
        if(isinstance(config,Config_Base_Agent)):
            self.batch_size = config.get_batch_size()
            self.clip_rewards = config.get_clip_rewards()
            self.architecture = config.get_architecture()
            self.input_dim = config.get_input_dim()
            self.output_size = config.get_output_size()
            self.is_mask_needed = config.get_is_mask_needed()
            self.random_episodes_to_run = config.get_random_episodes_to_run()
            self.epsilon_decay_rate_denominator = config.get_epsilon_decay_rate_denominator()
        else:
            self.batch_size = 1
            self.clip_rewards = False
            self.architecture = None
            self.input_dim = None 
            self.output_size = None
            self.is_mask_needed = False
            self.random_episodes_to_run = 0
            self.epsilon_decay_rate_denominator = 1


    def get_batch_size(self):
        if(self.batch_size == None):
            raise ValueError("Batch Size Not Defined")
        return self.batch_size
    
    def get_clip_rewards(self):
        if(self.clip_rewards == None):
            raise ValueError("Clip Rewards Not Defined")
        return self.clip_rewards

    def get_architecture(self):
        if(self.architecture == None):
            raise ValueError("Architecture Not Defined")
        return self.architecture
        
    def get_input_dim(self):
        if(self.input_dim != None):
            return self.input_dim
        else:
            return torch.tensor(torch.tensor([self.environment.reset()]).shape)
    
    def get_output_size(self):
        """ also adds batch dimension """
        if(self.output_size != None):
            return self.output_size
        try:
            return  self.environment.action_space.n
        except:
            raise ValueError("Output Size Not Defined and Can't Be figured out")

    def get_is_mask_needed(self):
        if(self.is_mask_needed != None):
            return self.is_mask_needed
        else:
            return True if self.get_environment().unwrapped.id in ['Gomoku'] else False

    def get_random_episodes_to_run(self):
        if(self.random_episodes_to_run != None):
            return self.random_episodes_to_run
        else:
            raise ValueError("Random Episodes To Run Not Defined")

    def get_epsilon_decay_rate_denominator(self):
        if(self.epsilon_decay_rate_denominator != None):
            return self.epsilon_decay_rate_denominator
        else:
            raise ValueError("Epsilon Decay Rate Denominator Not Defined")
        

class Base_Agent(object):


    prepared_games = ["CartPole","Gomoku"]

    def __init__(self, config: Config_Base_Agent):
        self.logger = self.setup_logger()
        self.debug_mode = config.get_debug_mode()
        # if self.debug_mode: self.tensorboard = SummaryWriter()
        self.config = config
        self.set_random_seeds(config.get_seed())
        self.environment = config.get_environment()
        self.environment_title = self.get_environment_title()
        if(self.environment_title not in self.prepared_games):
            raise ValueError("This game was not implemented")
        self.action_types = "DISCRETE" if self.environment.action_space.dtype == np.int64 else "CONTINUOUS"
        self.action_size = self.config.get_output_size()
        self.action_mask_required = self.config.get_is_mask_needed()
        self.input_shape = self.config.get_input_dim()
        self.lowest_possible_episode_score = self.get_lowest_possible_episode_score()
        self.average_score_required_to_win = self.get_score_required_to_win()
        self.rolling_score_window = self.get_trials()
        # self.max_steps_per_episode = self.environment.spec.max_episode_steps
        self.total_episode_score_so_far = 0
        self.game_full_episode_scores = []
        self.rolling_results = []
        self.max_rolling_score_seen = float("-inf")
        self.max_episode_score_seen = float("-inf")
        self.episode_number = 0
        self.device = "cuda:0" if config.get_use_GPU() else "cpu"
        self.visualise_results_boolean = config.visualise_individual_results
        self.global_step_number = 0
        self.turn_off_exploration = False
        gym.logger.set_level(40)  # stops it from printing an unnecessary warning
        self.log_game_info()
        self.writer = SummaryWriter()



    """ Methods to Manage Environment """
    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        self.environment.seed(self.config.get_seed())
        self.state = self.environment.reset()
        
        self.next_state = None
        self.action = None
        self.reward = None
        self.done = False
        self.total_episode_score_so_far = 0
        self.episode_states = []
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_next_states = []
        self.episode_dones = []
        self.episode_desired_goals = []
        self.episode_achieved_goals = []
        self.episode_observations = []
        if "exploration_strategy" in self.__dict__.keys(): self.exploration_strategy.reset()
        self.logger.info("Reseting game -- New start state {}".format(self.state))

    def conduct_action(self, action):
        """Conducts an action in the environment"""
        next_state, reward, done, _ = self.environment.step(action)
        self.set_next_state(next_state)
        self.set_reward(reward)
        self.set_done(done)
        self.total_episode_score_so_far += self.get_reward()
        if self.config.get_clip_rewards(): self.set_reward( max(min(self.get_reward(), 1.0), -1.0))
        if(self.done == True):
            self.logger.info("Game ended -- Final state {}".format(self.get_next_state()))
            self.logger.info("reward: {}".format(self.get_reward()))

    def get_environment_title(self):
        """Extracts name of environment from it"""
        try:
            name = self.environment.unwrapped.id
        except AttributeError:
            try:
                if str(self.environment.unwrapped)[1:11] == "FetchReach": return "FetchReach"
                elif str(self.environment.unwrapped)[1:8] == "AntMaze": return "AntMaze"
                elif str(self.environment.unwrapped)[1:7] == "Hopper": return "Hopper"
                elif str(self.environment.unwrapped)[1:9] == "Walker2d": return "Walker2d"
                else:
                    name = self.environment.spec.id.split("-")[0]
            except AttributeError:
                name = str(self.environment.env)
                if name[0:10] == "TimeLimit<": name = name[10:]
                name = name.split(" ")[0]
                if name[0] == "<": name = name[1:]
                if name[-3:] == "Env": name = name[:-3]
        return name

    def get_score_required_to_win(self):
        """Gets average score required to win game"""
        print("TITLE ", self.environment_title)
        if self.environment_title == "FetchReach": return -5
        if self.environment_title in ["AntMaze", "Hopper", "Walker2d","Gomoku"]:
            print("Score required to win set to infinity therefore no learning rate annealing will happen")
            return float("inf")
        try: return self.environment.unwrapped.reward_threshold
        except AttributeError:
            try:
                return self.environment.spec.reward_threshold
            except AttributeError:
                return self.environment.unwrapped.spec.reward_threshold

    def get_lowest_possible_episode_score(self):
        """Returns the lowest possible episode score you can get in an environment"""
        if self.environment_title == "Taxi": return -800
        return None

    def get_trials(self):
        """Gets the number of trials to average a score over"""
        if self.environment_title in ["AntMaze", "FetchReach", "Hopper", "Walker2d", "CartPole","Gomoku"]: return 100
        try: return self.environment.unwrapped.trials
        except AttributeError: return self.environment.spec.trials

    def get_action_mask(self):
        if(self.action_mask_required == True):
            return torch.tensor(self.environment.get_mask())
        else:
            return None

    def step(self):
        """Takes a step in the game. This method must be overriden by any agent"""
        raise ValueError("Step needs to be implemented by the agent")

    '''
    def create_NN(self, input_dim, output_dim, key_to_use=None, override_seed=None, hyperparameters=None):
        """Creates a neural network for the agents to use"""
        if hyperparameters is None: hyperparameters = self.hyperparameters
        if key_to_use: hyperparameters = hyperparameters[key_to_use]
        if override_seed: seed = override_seed
        else: seed = self.config.get_seed()

        default_hyperparameter_choices = {"output_activation": None, "hidden_activations": "relu", "dropout": 0.0,
                                          "initialiser": "default", "batch_norm": False,
                                          "columns_of_data_to_be_embedded": [],
                                          "embedding_dimensions": [], "y_range": ()}

        for key in default_hyperparameter_choices:
            if key not in hyperparameters.keys():
                hyperparameters[key] = default_hyperparameter_choices[key]

        return NN(input_dim=input_dim, layers_info=hyperparameters["linear_hidden_units"] + [output_dim],
                  output_activation=hyperparameters["final_layer_activation"],
                  batch_norm=hyperparameters["batch_norm"], dropout=hyperparameters["dropout"],
                  hidden_activations=hyperparameters["hidden_activations"], initialiser=hyperparameters["initialiser"],
                  columns_of_data_to_be_embedded=hyperparameters["columns_of_data_to_be_embedded"],
                  embedding_dimensions=hyperparameters["embedding_dimensions"], y_range=hyperparameters["y_range"],
                  random_seed=seed).to(self.device)
    '''

    def create_NN_through_NNbuilder(self,input_dim,output_size,smoothing):
        return NNbuilder(architecture=self.config.architecture,input_dim=input_dim,output_size=output_size,smoothing=smoothing).to(self.device)

    def run_n_episodes(self, num_episodes=None, show_whether_achieved_goal=True, save_and_print_results=True):
        """Runs game to completion n times and then summarises results and saves model (if asked to)"""
        if num_episodes is None: num_episodes = self.config.get_num_episodes_to_run()
        start = time.time()
        while self.episode_number < num_episodes:
            self.reset_game()
            self.step()
            if save_and_print_results: self.save_and_print_result()
        time_taken = time.time() - start
        if show_whether_achieved_goal: self.show_whether_achieved_goal()
        if self.config.get_save_model(): self.locally_save_policy()
        self.writer.close()
        return self.game_full_episode_scores, self.rolling_results, time_taken

    def turn_on_any_epsilon_greedy_exploration(self):
        """Turns off all exploration with respect to the epsilon greedy exploration strategy"""
        print("Turning on epsilon greedy exploration")
        self.turn_off_exploration = False

    def turn_off_any_epsilon_greedy_exploration(self):
        """Turns off all exploration with respect to the epsilon greedy exploration strategy"""
        print("Turning off epsilon greedy exploration")
        self.turn_off_exploration = True

    def take_optimisation_step(self, optimizer, network, loss, clipping_norm=None, retain_graph=False):
        """Takes an optimisation step by calculating gradients given the loss and then updating the parameters"""
        if not isinstance(network, list): network = [network]
        optimizer.zero_grad() #reset gradients to 0
        loss.backward(retain_graph=retain_graph) #this calculates the gradients
        self.logger.info("Loss -- {}".format(loss.item()))
        if self.debug_mode: self.log_gradient_and_weight_information(network, optimizer)
        if clipping_norm is not None:
            for net in network:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clipping_norm) #clip gradients to help stabilise training
        optimizer.step() #this applies the gradients

    def soft_update_of_target_network(self, local_model, target_model, tau):
        """Updates the target network in the direction of the local network but by taking a step size
        less than one so the target network's parameter values trail the local networks. This helps stabilise training"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def update_learning_rate(self, starting_lr,  optimizer):
        """Lowers the learning rate according to how close we are to the solution"""
        if len(self.rolling_results) > 0:
            last_rolling_score = self.rolling_results[-1]
            if last_rolling_score > 0.75 * self.average_score_required_to_win:
                new_lr = starting_lr / 100.0
            elif last_rolling_score > 0.6 * self.average_score_required_to_win:
                new_lr = starting_lr / 20.0
            elif last_rolling_score > 0.5 * self.average_score_required_to_win:
                new_lr = starting_lr / 10.0
            elif last_rolling_score > 0.25 * self.average_score_required_to_win:
                new_lr = starting_lr / 2.0
            else:
                new_lr = starting_lr
            for g in optimizer.param_groups:
                g['lr'] = new_lr
        if random.random() < 0.001: self.logger.info("Learning rate {}".format(new_lr))

    def freeze_all_but_output_layers(self, network):
        """Freezes all layers except the output layer of a network"""
        print("Freezing hidden layers")
        for param in network.named_parameters():
            param_name = param[0]
            assert "hidden" in param_name or "output" in param_name or "embedding" in param_name, "Name {} of network layers not understood".format(param_name)
            if "output" not in param_name:
                param[1].requires_grad = False

    def unfreeze_all_layers(self, network):
        """Unfreezes all layers of a network"""
        print("Unfreezing all layers")
        for param in network.parameters():
            param.requires_grad = True

    def save_experience(self, memory=None, experience=None):
        """Saves the recent experience to the memory buffer"""
        if memory is None: memory = self.memory
        if experience is None: experience = self.state, self.action, self.reward, self.next_state, self.done
        memory.add_experience(*experience)

    def enough_experiences_to_learn_from(self):
        """Boolean indicated whether there are enough experiences in the memory buffer to learn from"""
        return len(self.memory) > self.config.get_batch_size()

     
    """ Managing information """
    def setup_logger(self):
        """Sets up the logger"""
        filename = "Training.log"
        try:
            if os.path.isfile(filename):
                os.remove(filename)
        except: pass

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        # create a file handler
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.INFO)
        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(handler)
        return logger

    def log_game_info(self):
        """Logs info relating to the game"""
        for ix, param in enumerate([self.environment_title, self.action_types, self.action_size, self.lowest_possible_episode_score,
                      self.input_shape, self.config.get_architecture(), self.average_score_required_to_win, self.rolling_score_window,
                      self.device]):
            self.logger.info("{} -- {}".format(ix, param))

    def set_random_seeds(self, random_seed):
        """Sets all possible random seeds so results can be reproduced"""
        os.environ['PYTHONHASHSEED'] = str(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(random_seed)
        # tf.set_random_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
            torch.cuda.manual_seed(random_seed)
        if hasattr(gym.spaces, 'prng'):
            gym.spaces.prng.seed(random_seed)

    def track_episodes_data(self):
        """Saves the data from the recent episodes"""
        self.episode_states.append(self.state)
        self.episode_actions.append(self.action)
        self.episode_rewards.append(self.reward)
        self.episode_next_states.append(self.next_state)
        self.episode_dones.append(self.done)

    def save_and_print_result(self):
        """Saves and prints results of the game"""
        self.save_result()
        self.print_rolling_result()

    def save_result(self):
        """Saves the result of an episode of the game"""
        self.game_full_episode_scores.append(self.total_episode_score_so_far)
        self.rolling_results.append(np.mean(self.game_full_episode_scores[-1 * self.rolling_score_window:]))
        self.save_max_result_seen()

    def save_max_result_seen(self):
        """Updates the best episode result seen so far"""
        if self.game_full_episode_scores[-1] > self.max_episode_score_seen:
            self.max_episode_score_seen = self.game_full_episode_scores[-1]

        if self.rolling_results[-1] > self.max_rolling_score_seen:
            if len(self.rolling_results) > self.rolling_score_window:
                self.max_rolling_score_seen = self.rolling_results[-1]

    def print_rolling_result(self):
        """Prints out the latest episode results"""
        text = """"\r Episode {0}, Score: {3: .2f}, Max score seen: {4: .2f}, Rolling score: {1: .2f}, Max rolling score seen: {2: .2f}, avg_score: {5: 2.2f}"""
        sys.stdout.write(text.format(len(self.game_full_episode_scores), self.rolling_results[-1], self.max_rolling_score_seen,
                                     self.game_full_episode_scores[-1], self.max_episode_score_seen, sum(self.game_full_episode_scores)/len(self.game_full_episode_scores)))
        sys.stdout.flush()

    def show_whether_achieved_goal(self):
        """Prints out whether the agent achieved the environment target goal"""
        index_achieved_goal = self.achieved_required_score_at_index()
        print(" ")
        if index_achieved_goal == -1: #this means agent never achieved goal
            print("\033[91m" + "\033[1m" +
                  "{} did not achieve required score \n".format(self.agent_name) +
                  "\033[0m" + "\033[0m")
        else:
            print("\033[92m" + "\033[1m" +
                  "{} achieved required score at episode {} \n".format(self.agent_name, index_achieved_goal) +
                  "\033[0m" + "\033[0m")

    def achieved_required_score_at_index(self):
        """Returns the episode at which agent achieved goal or -1 if it never achieved it"""
        for ix, score in enumerate(self.rolling_results):
            if score > self.average_score_required_to_win:
                return ix
        return -1

    def log_gradient_and_weight_information(self, network, optimizer):

        # log weight information
        total_norm = 0
        for name, param in network.named_parameters():
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        self.logger.info("Gradient Norm {}".format(total_norm))

        for g in optimizer.param_groups:
            learning_rate = g['lr']
            break
        self.logger.info("Learning Rate {}".format(learning_rate))


    """ setters """
    def set_state(self,state):
        self.state = state
        self.store_state(state)
    
    def set_next_state(self,next_state):
        self.next_state = next_state
        self.store_next_state(next_state)

    def set_action(self,action):
        self.action = action
        self.store_action(action)

    def set_reward(self,reward):
        self.reward = reward
        self.store_reward(reward)

    def set_done(self,done):
        self.done = done
        self.store_done(done)   


    """ getters """
    def get_state(self):
        return self.state 
    
    def get_next_state(self):
        return self.next_state

    def get_action(self):
        return self.action

    def get_reward(self):
        return self.reward

    def get_done(self):
        return self.done 


    """ storage in lists """
    def store_state(self,state):
        self.episode_states.append(state)
    
    def store_reward(self,reward):
        self.episode_rewards.append(reward)

    def store_action(self,action):
        self.episode_actions.append(action)

    def store_next_state(self,next_state):
        self.episode_next_states.append(next_state)
    
    def store_done(self,done):
        self.episode_dones.append(done)
    
    def store_desired_goal(self,desired_goal):
        self.episode_desired_goals.append(desired_goal)

    def store_achieved_goal(self,achieved_goal):
        self.episode_achieved_goals.append(achieved_goal)
    
    def store_observation(self,observation):
        self.episode_observations.append(observation)


    """ get in lists """
    def get_episode_states(self):
        return self.episode_states

    def get_episode_actions(self):
        return self.episode_actions

    def get_episode_rewards(self):
        return self.episode_rewards
        
    def get_episode_next_states(self):
        return self.episode_next_states

    def get_episode_dones(self):
        return self.episode_dones

    def get_episode_desired_goals(self):
        return self.episode_desired_goals

    def get_episode_achieved_goals(self):
        return self.episode_achieved_goals

    def get_episode_observations(self):
        return self.episode_observations

    """ Other """
    @staticmethod
    def move_gradients_one_model_to_another(from_model, to_model, set_from_gradients_to_zero=False):
        """Copies gradients from from_model to to_model"""
        for from_model, to_model in zip(from_model.parameters(), to_model.parameters()):
            to_model._grad = from_model.grad.clone()
            if set_from_gradients_to_zero: from_model._grad = None

    @staticmethod
    def copy_model_over(from_model, to_model):
        """Copies model parameters from from_model to to_model"""
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            to_model.data.copy_(from_model.data.clone())
