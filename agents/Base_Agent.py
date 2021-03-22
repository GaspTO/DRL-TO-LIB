import logging
import os
import sys
import gym
import random
import numpy as np
import torch
import time
from torch.utils.tensorboard import SummaryWriter
from torch.optim import optimizer
from utilities.data_structures.Config import Config
from agents.Agent import Agent



class Config_Base_Agent(Config):
    def __init__(self,config=None):
        Config.__init__(self,config)
        if(isinstance(config,Config_Base_Agent)):
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
        

class Base_Agent(Agent):
    prepared_games = ["Gomoku","K_Row"]
    def __init__(self, config: Config_Base_Agent):
        self.setup_logger()
        self.debug_mode = config.get_debug_mode()
        self.writer = SummaryWriter()
        self.config = config
        self.set_random_seeds(config.get_seed())
        self.environment = config.get_environment()
        self.environment_title = self.get_environment_title()
        if(self.environment_title not in self.prepared_games):
            raise ValueError("This game was not implemented")
        self.action_types = "dumb..." #todo fix me
        #self.action_types = "DISCRETE" if self.environment.action_space.dtype == np.int64 else "CONTINUOUS"
        self.action_size = self.config.get_output_size() #todo fixme
        self.action_mask_required = self.config.get_is_mask_needed()
        self.input_shape = 3 #todo fixme 
        #self.input_shape = self.config.get_input_dim() #todo fixme
        self.average_score_required_to_win = self.get_score_required_to_win()
        self.rolling_score_window = self.get_trials()
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
        

    """ Main Methods """
    def run_n_episodes(self, num_episodes=None, show_whether_achieved_goal=True, save_and_print_results=True):
        """Runs game to completion n times and then summarises results and saves model (if asked to)"""
        if num_episodes is None: num_episodes = self.config.get_num_episodes_to_run()
        start = time.time()
        while self.episode_number < num_episodes:
            self.reset_game()
            self.step()
            if self.debug_mode == True: self.logger.info("Game ended -- State and Reward Sequence is:\n{}".format(self.pack_states_and_rewards_side_by_side()))
            else: self.logger.info("Game ended -- Last State:\n{}".format(self.episode_states[-1]))
            if save_and_print_results: self.save_and_print_result()
            self.writer.flush()
        time_taken = time.time() - start
        if show_whether_achieved_goal: self.show_whether_achieved_goal()
        if self.config.get_save_model(): self.locally_save_policy()
        self.writer.close()
        return self.game_full_episode_scores, self.rolling_results, time_taken

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        self.episode_states = []
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_next_states = []
        self.episode_dones = []
        self.episode_probabilities = []
        self.episode_desired_goals = []
        self.episode_achieved_goals = []
        self.episode_observations = []
        self.environment.seed(self.config.get_seed())
        self.state = self.environment.reset()
        self.next_state = None
        self.action = None
        self.reward = None
        self.done = False
        self.total_episode_score_so_far = 0
        if "exploration_strategy" in self.__dict__.keys(): self.exploration_strategy.reset()
        self.logger.info("Reseting game \n*\n*\n*\n*")

    def step(self):
        """Takes a step in the game. This method must be overriden by any agent"""
        raise ValueError("Step needs to be implemented by the agent")

    def conduct_action(self, action):
        """Conducts an action in the environment"""
        next_state, reward, done, _ = self.environment.step(action)
        self.episode_actions.append(action)
        self.next_state = next_state
        self.episode_next_states.append(self.next_state)
        self.reward = reward
        self.episode_rewards.append(self.reward)
        self.done = done
        self.episode_dones.append(self.done)
        self.total_episode_score_so_far += self.reward
        if self.config.get_clip_rewards(): self.reward =  max(min(self.reward, 1.0), -1.0)
        if(self.done == True):
            self.logger.info("final_reward: {}".format(self.reward))

    def get_action_mask(self):
        if(self.action_mask_required == True):
            return torch.tensor(self.environment.get_mask())
        else:
            return None
    
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
                try:
                    name = self.environment.get_id()
                except:
                    name = str(self.environment.env)
                    if name[0:10] == "TimeLimit<": name = name[10:]
                    name = name.split(" ")[0]
                    if name[0] == "<": name = name[1:]
                    if name[-3:] == "Env": name = name[:-3]
        return name

    def get_score_required_to_win(self):
        """Gets average score required to win game"""
        if self.environment_title == "FetchReach": return -5
        if self.environment_title in ["AntMaze", "Hopper", "Walker2d","Gomoku","K_Row"]:
            #print("Score required to win set to infinity therefore no learning rate annealing will happen")
            return float("inf")
        try: return self.environment.unwrapped.reward_threshold
        except AttributeError:
            try:
                return self.environment.spec.reward_threshold
            except AttributeError:
                return self.environment.unwrapped.spec.reward_threshold

    def get_trials(self):
        """Gets the number of trials to average a score over"""
        if self.environment_title in ["AntMaze", "FetchReach", "Hopper", "Walker2d", "CartPole","Gomoku","K_Row"]: return 100
        try: return self.environment.unwrapped.trials
        except AttributeError: return self.environment.spec.trials
    
    def turn_on_any_epsilon_greedy_exploration(self):
        """Turns off all exploration with respect to the epsilon greedy exploration strategy"""
        print("Turning on epsilon greedy exploration")
        self.turn_off_exploration = False

    def turn_off_any_epsilon_greedy_exploration(self):
        """Turns off all exploration with respect to the epsilon greedy exploration strategy"""
        print("Turning off epsilon greedy exploration")
        self.turn_off_exploration = True

    
    """ Information """
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
        self.logger = logger

    def log_game_info(self):
        """Logs info relating to the game"""
        for ix, param in enumerate([self.environment_title, self.action_types, self.action_size, 
                      self.input_shape, self.config.get_architecture(), self.average_score_required_to_win, self.rolling_score_window,
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

    def pack_states_and_rewards_side_by_side(self):
        height = self.episode_states[0].shape[0]        
        output = []
        for line_no in range(height):
            for state in self.episode_states:
                state_line = str(list(state[line_no]))
                output.append(state_line)
            #state_line = str(list(self.next_state[line_no]))
            #output.append(state_line)
            output.append("\n")
        for rew_no in range(len(self.episode_rewards)):
            rew_str = str(self.episode_rewards[rew_no])
            output.append(" " * (len(output[rew_no])-len(rew_str)) + rew_str)
        return ''.join(output)

    def achieved_required_score_at_index(self):
        """Returns the episode at which agent achieved goal or -1 if it never achieved it"""
        for ix, score in enumerate(self.rolling_results):
            if score > self.average_score_required_to_win:
                return ix
        return -1

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


    """ Replay Memory Storage Methods"""
    def sample_transitions(self):
        """Draws a random sample of transitions from the memory buffer"""
        transitions = self.memory.sample()
        states, actions, rewards, next_states, dones = transitions
        return states, actions, rewards, next_states, dones

    def save_transition(self, memory=None, transition=None):
        """Saves the recent transition to the memory buffer"""
        if memory is None: memory = self.memory
        if transition is None: transition = self.state, self.action, self.reward, self.next_state, self.done
        memory.add_transition(*transition)

    def enough_transitions_to_learn_from(self):
        """Boolean indicated whether there are enough transitions in the memory buffer to learn from"""
        return len(self.memory) > self.config.get_batch_size()


    """ Other """
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

    