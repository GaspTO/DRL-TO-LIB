#import logging
from abc import abstractclassmethod, abstractmethod
from utilities.logger import logger
import os
import sys
import gym
import random
import numpy as np
import torch
import time
import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.optim import optimizer
from utilities.data_structures.Config import Config
from agents.Agent import Agent



class Config_Learning_Agent(Config):
    def __init__(self,config=None):
        Config.__init__(self,config)
        if(isinstance(config,Config_Learning_Agent)):
            self.batch_size = config.get_batch_size()
            self.gradient_clipping_norm = config.get_gradient_clipping_norm()
            self.clip_rewards = config.get_clip_rewards()
            self.architecture = config.get_architecture()
            #self.input_dim = config.get_input_dim()
            #self.output_size = config.get_output_size()
            self.is_mask_needed = config.get_is_mask_needed()
            self.random_episodes_to_run = config.get_random_episodes_to_run()
            self.epsilon_decay_rate_denominator = config.get_epsilon_decay_rate_denominator()
        else:
            self.batch_size = 1
            self.gradient_clipping_norm = 0.7
            self.clip_rewards = False
            self.architecture = None
            #self.input_dim = None 
            #self.output_size = None
            self.is_mask_needed = False
            self.random_episodes_to_run = 0
            self.epsilon_decay_rate_denominator = 1

    def get_batch_size(self):
        if(self.batch_size == None):
            raise ValueError("Batch Size Not Defined")
        return self.batch_size
    
    def get_gradient_clipping_norm(self):
        return self.gradient_clipping_norm

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
        return 3 #todo fixme
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
        

class Learning_Agent(Agent):
    prepared_games = ["Gomoku","K_Row"]
    def __init__(self, config: Config_Learning_Agent):
        self.setup_logger()
        self.debug_mode = config.get_debug_mode()
        self.writer = SummaryWriter("logs/runs")
        self.config = config
        self.set_random_seeds(config.get_seed())
        self.environment = config.get_environment()
        self.environment_title = self.get_environment_title()
        #if(self.environment_title not in self.prepared_games):
        #    raise ValueError("This game was not implemented")
        self.action_types = "dumb..." #todo fix me
        #self.action_types = "DISCRETE" if self.environment.action_space.dtype == np.int64 else "CONTINUOUS"
        self.action_size = self.config.get_output_size() #todo fixme
        self.input_shape = 3 #todo fixme 
        #self.input_shape = self.config.get_input_dim() #todo fixme
        self.rolling_score_window = 100
        self.total_episode_score_so_far = 0
        self.game_full_episode_scores = []
        self.rolling_results = []
        self.max_rolling_score_seen = float("-inf")
        self.max_episode_score_seen = float("-inf")
        self.episode_number = 0
        self.device = "cuda" if config.get_use_GPU() else "cpu"
        self.visualise_results_boolean = config.visualise_individual_results
        self.global_step_number = 0
        self.turn_off_exploration = False
        gym.logger.set_level(40)  
        self.log_game_info()


        
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    *                            MAIN INTERFACE                               
    *            Main interface to be used by every implemented agent               
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    @abstractmethod
    def play(self,observations:np.array=None,policy=None,info=None) -> tuple([np.array,dict]):
        return NotImplementedError

    def run_n_episodes(self, num_episodes=None, show_whether_achieved_goal=True, save_and_print_results=True):
        """Runs game to completion n times and then summarises results and saves model (if asked to)"""
        if num_episodes is None: num_episodes = self.config.get_num_episodes_to_run()
        start = time.time()
        while self.episode_number < num_episodes:
            self.reset()
            self.do_episode()
            if self.debug_mode == True: self.logger.info("Game ended -- Observation and Reward Sequence is:\n{}".format(self.pack_observations_and_rewards_side_by_side()))
            else: self.logger.info("Game ended -- Last observation:\n{}".format(self.episode_next_observations[-1]))
            if save_and_print_results: self.save_and_print_result()
            self.writer.flush()
        time_taken = time.time() - start
        if show_whether_achieved_goal: self.show_whether_achieved_goal()
        if self.config.get_save_model(): self.locally_save_policy()
        self.writer.close()
        return self.game_full_episode_scores, self.rolling_results, time_taken

    def do_episode(self):
        while not self.is_episode_finished():
            self.step()
            if self.time_to_learn():
                self.before_learn_block()
                self.learn()
                self.after_learn_block()
            self.save_step_info()
            self.end_step_block() 
            self.advance_to_next_state()
        self.end_episode()

    def is_episode_finished(self):
        return self.done

    @abstractmethod
    def step(self):
        return NotImplementedError

    def time_to_learn(self):
        return self.done

    def before_learn_block(self):
        return

    @abstractmethod
    def learn():
        return NotImplementedError

    def after_learn_block(self):
        return

    def save_step_info(self):
        self.episode_observations.append(self.observation)
        self.episode_masks.append(self.mask)
        self.episode_actions.append(self.action)
        self.episode_rewards.append(self.reward)
        self.episode_next_observations.append(self.next_observation)
        self.episode_dones.append(self.done)
        self.total_episode_score_so_far += self.reward
        self.episode_step_number += 1

    def end_step_block(self):
        return

    def advance_to_next_state(self):
        self.observation = self.next_observation
        self.mask = self.environment.get_mask()

    def end_episode(self):
        self.episode_number += 1
        if self.debug_mode:
            self.log_updated_probabilities()
        self.logger.info("total reward: {}".format(self.total_episode_score_so_far))

    def reset(self):
        self.logger.info("Reseting game \n*\n*\n*\n*")
        super().reset()
        self.environment.seed(self.config.get_seed())
        self.total_episode_score_so_far = 0
        self.episode_step_number = 0
        self.episode_observations = []
        self.episode_masks = []
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_next_observations = []
        self.episode_dones = []
        self.episode_probabilities = []
        self.episode_action_probabilities = []
        self.episode_action_log_probabilities = []
        self.episode_desired_goals = []
        self.episode_achieved_goals = []
        self.episode_observations = []
        if "exploration_strategy" in self.__dict__.keys(): self.exploration_strategy.reset()
        



    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    *                            AUXILIARY METHODS 
    * Methods that are often used by several agents, although not necessary in
    every agents
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    ''' policy '''
    def take_optimisation_step(self, optimizer, network, loss, clipping_norm=None, retain_graph=False):
        """Takes an optimisation step by calculating gradients given the loss and then updating the parameters"""
        optimizer.zero_grad() #reset gradients to 0
        loss.backward(retain_graph=retain_graph) #this calculates the gradients
        self.logger.info("Total Loss -- {}".format(loss.item()))
        self.writer.add_scalar("optimisation_loss",loss,len(self.game_full_episode_scores))
        if self.debug_mode: self.log_gradient_and_weight_information(network, optimizer)
        if clipping_norm is not None:
                torch.nn.utils.clip_grad_norm_(network.parameters(), clipping_norm) #clip gradients to help stabilise training
        optimizer.step() #this applies the gradients

    def soft_update_of_target_network(self, local_model, target_model, tau):
        """Updates the target network in the direction of the local network but by taking a step size
        less than one so the target network's parameter values trail the local networks. This helps stabilise training"""
        if(local_model == target_model): raise ValueError("Can't update target model if it's the same as local model")
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

    
    ''' Helpful '''
    def get_environment_title(self):
        """Extracts name of environment from it"""
        return self.environment.get_name()

    def turn_on_any_epsilon_greedy_exploration(self):
        """Turns off all exploration with respect to the epsilon greedy exploration strategy"""
        print("Turning on epsilon greedy exploration")
        self.turn_off_exploration = False

    def turn_off_any_epsilon_greedy_exploration(self):
        """Turns off all exploration with respect to the epsilon greedy exploration strategy"""
        print("Turning off epsilon greedy exploration")
        self.turn_off_exploration = True


    ''' Replay Memory Storage Methods '''
    def sample_transitions(self):
        """Draws a random sample of transitions from the memory buffer"""
        transitions = self.memory.sample()
        observation, actions, rewards, next_observation, dones = transitions
        return observation, actions, rewards, next_observation, dones

    def save_transition(self, memory=None, transition=None):
        """Saves the recent transition to the memory buffer"""
        if memory is None: memory = self.memory
        if transition is None: transition = self.observation, self.action, self.reward, self.next_observation, self.done
        memory.add_transition(*transition)

    def enough_transitions_to_learn_from(self):
        """Boolean indicated whether there are enough transitions in the memory buffer to learn from"""
        return len(self.memory) > self.config.get_batch_size()


    ''' Logging '''
    def setup_logger(self):
        self.logger = logger

    def log_game_info(self):
        """Logs info relating to the game"""
        for ix, param in enumerate([self.environment_title, self.action_types, self.action_size, 
                      self.input_shape, self.config.get_architecture(), self.rolling_score_window,
                      self.device]):
            self.logger.info("{} -- {}".format(ix, param))

    def log_gradient_and_weight_information(self, network, optimizer):
        # log weight information
        total_norm = 0
        for name, param in network.named_parameters():
            if(param.grad is not None):
                param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        for g in optimizer.param_groups:
            learning_rate = g['lr']
            break
        self.logger.info("Gradient Norm {}, Learning Rate {}".format(total_norm,learning_rate))

    def save_and_print_result(self):
        """Saves and prints results of the game"""
        self.save_result()
        self.print_and_log_rolling_result()

    def print_and_log_rolling_result(self):
        """Prints out the latest episode results"""
        text = """Episode {0}, Score: {3: .2f}, Max score seen: {4: .2f}, Rolling score: {1: .2f}, Max rolling score seen: {2: .2f},"""
        formatted_text = text.format(len(self.game_full_episode_scores), self.rolling_results[-1], self.max_rolling_score_seen,
                                     self.game_full_episode_scores[-1], self.max_episode_score_seen)
        sys.stdout.write("\r" + formatted_text)
        sys.stdout.flush()   
        self.logger.info(formatted_text)
        self.writer.add_scalar("rolling_results",self.rolling_results[-1],len(self.game_full_episode_scores))

    def save_max_result_seen(self):
        """Updates the best episode result seen so far"""
        if self.game_full_episode_scores[-1] > self.max_episode_score_seen:
            self.max_episode_score_seen = self.game_full_episode_scores[-1]

        if self.rolling_results[-1] > self.max_rolling_score_seen:
            if len(self.rolling_results) > self.rolling_score_window:
                self.max_rolling_score_seen = self.rolling_results[-1]

    def pack_observations_and_rewards_side_by_side(self):
        height = self.episode_observations[0].shape[0]        
        output = []
        for line_no in range(height):
            for observation in self.episode_observations:
                observation_line = str(list(observation[line_no]))
                output.append(observation_line)
            output.append("\n")
        for rew_no in range(len(self.episode_rewards)):
            rew_str = str(self.episode_rewards[rew_no])
            output.append(" " * (len(output[rew_no])-len(rew_str)) + rew_str)
        return ''.join(output)

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

    def locally_save_policy(self):
        """Saves the policy"""
        torch.save(self.q_network_local.state_dict(), "Models/{}_local_network.pt".format(self.agent_name))

    def save_result(self):
        """Saves the result of an episode of the game"""
        self.game_full_episode_scores.append(self.total_episode_score_so_far)
        self.rolling_results.append(np.mean(self.game_full_episode_scores[-1 * self.rolling_score_window:]))
        self.save_max_result_seen()


    ''' Other '''
    def clone(self):
        raise NotImplementedError

    def __del__(self):
        self.writer.close()

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

    