import copy
import random
import time
import numpy as np
import torch
from torch import multiprocessing
from torch.multiprocessing import Queue
from torch.optim import Adam
from agents.Base_Agent import Base_Agent, Config_Base_Agent
from utilities.Utility_Functions import create_actor_distribution, SharedAdam



class Config_A3C(Config_Base_Agent):
    def __init__(self,config=None):
        Config_Base_Agent.__init__(self,config)
        if(isinstance(config,Config_A3C)):
            self.discount_rate = config.get_discount_rate()
            self.epsilon_decay_rate_denominator = config.get_epsilon_decay_rate_denominator()
            self.exploration_worker_difference = config.get_exploration_worker_difference()
            self.gradient_clipping_norm = config.get_gradient_clipping_norm()
            self.learning_rate = config.get_learning_rate()
            self.normalise_rewards = config.get_normalise_rewards()
        else:
            self.discount_rate = 0.99
            self.epsilon_decay_rate_denominator = 1
            self.exploration_worker_difference = 2.0
            self.gradient_clipping_norm = 0.7
            self.learning_rate = 1
            self.normalise_rewards = True

    def get_discount_rate(self):
        return self.discount_rate

    def get_epsilon_decay_rate_denominator(self):
        return self.epsilon_decay_rate_denominator

    def get_exploration_worker_difference(self):
        return self.exploration_worker_difference

    def get_gradient_clipping_norm(self):
        return self.gradient_clipping_norm

    def get_normalise_rewards(self):
        return self.normalise_rewards
        
class A3C(Base_Agent):
    """Actor critic A3C algorithm from deepmind paper https://arxiv.org/pdf/1602.01783.pdf"""
    agent_name = "A3C"
    def __init__(self, config: Config_A3C):
        super(A3C, self).__init__(config)
        self.num_processes = multiprocessing.cpu_count()
        self.worker_processes = max(1, self.num_processes - 2)
        print("...crapp")
        self.worker_processes = 2
        self.actor_critic = self.create_NN_through_NNbuilder(input_dim=self.input_shape, output_size=self.action_size + 1, smoothing=0.001)
        self.actor_critic_optimizer = SharedAdam(self.actor_critic.parameters(), lr=config.learning_rate, eps=1e-4)

    def run_n_episodes(self, num_episodes = None):
        """Runs game to completion n times and then summarises results and saves model (if asked to)"""
        start = time.time()
        results_queue = Queue()
        gradient_updates_queue = Queue()
        episode_number = multiprocessing.Value('i', 0)
        self.optimizer_lock = multiprocessing.Lock()
        if(num_episodes != None):
            self.num_episodes_to_run = num_episodes
        else:
            self.num_episodes_to_run = self.config.num_episodes_to_run
        episodes_per_process = int(self.num_episodes_to_run / self.worker_processes) + 1
        processes = []
        self.actor_critic.share_memory()
        self.actor_critic_optimizer.share_memory()

        optimizer_worker = multiprocessing.Process(target=self.update_shared_model, args=(gradient_updates_queue,))
        optimizer_worker.start()

        for process_num in range(self.worker_processes):
            worker = Actor_Critic_Worker(process_num, copy.deepcopy(self.environment), self.actor_critic, episode_number, self.optimizer_lock,
                                    self.actor_critic_optimizer, self.config, episodes_per_process,
                                    self.config.epsilon_decay_rate_denominator,
                                    self.action_mask_required, 
                                    self.action_size, self.action_types,
                                    results_queue, copy.deepcopy(self.actor_critic), gradient_updates_queue)
            worker.start()
            processes.append(worker)
        self.print_results(episode_number, results_queue)
        for worker in processes:
            worker.join()
        optimizer_worker.terminate()
        time_taken = time.time() - start
        return self.game_full_episode_scores, self.rolling_results, time_taken

    def print_results(self, episode_number, results_queue):
        """Worker that prints out results as they get put into a queue"""
        while True:
            with episode_number.get_lock():
                carry_on = episode_number.value < self.config.num_episodes_to_run
            if carry_on:
                if not results_queue.empty():
                    self.total_episode_score_so_far = results_queue.get()
                    self.save_and_print_result()
            else: break

    def update_shared_model(self, gradient_updates_queue):
        """Worker that updates the shared model with gradients as they get put into the queue"""
        while True:          
            try:
                gradients = gradient_updates_queue.get()
            except:
                pass
            
            with self.optimizer_lock:
                self.actor_critic_optimizer.zero_grad()
                for grads, params in zip(gradients, self.actor_critic.parameters()):
                    params._grad = grads  # maybe need to do grads.clone()
                self.actor_critic_optimizer.step()
        

class Actor_Critic_Worker(torch.multiprocessing.Process,Base_Agent):

    """Actor critic worker that will play the game for the designated number of episodes """
    def __init__(self, worker_num, environment, shared_model, counter, optimizer_lock, shared_optimizer,
                 config, episodes_to_run, epsilon_decay_denominator, action_mask_required, action_size, action_types, results_queue,
                 local_model, gradient_updates_queue):
        torch.multiprocessing.Process.__init__(self)
        Base_Agent.__init__(self,config)
        self.environment = environment
        self.worker_num = worker_num

        self.gradient_clipping_norm = self.config.get_gradient_clipping_norm()
        self.discount_rate = self.config.discount_rate
        self.normalise_rewards = self.config.normalise_rewards

        self.set_seeds(self.worker_num)
        self.shared_model = shared_model
        self.local_model = local_model
        self.local_optimizer = Adam(self.local_model.parameters(), lr=0.0, eps=1e-4)
        self.counter = counter
        self.optimizer_lock = optimizer_lock
        self.shared_optimizer = shared_optimizer
        self.episodes_to_run = episodes_to_run
        self.epsilon_decay_denominator = epsilon_decay_denominator
        self.exploration_worker_difference = self.config.exploration_worker_difference
        self.action_types = action_types
        self.results_queue = results_queue
        self.episode_number = 0

        self.gradient_updates_queue = gradient_updates_queue


    """ Main """
    def run(self):
        self.step()

    def step(self):
        """Starts the worker"""
        torch.set_num_threads(1)
        for ep_ix in range(self.episodes_to_run):
            with self.optimizer_lock:
                self.copy_model_over(self.shared_model, self.local_model)
            epsilon_exploration = self.calculate_new_exploration()
            self.reset_game()

            while not self.done: #TODO in the A3C paper we don't have to complete a whole episode
                action, action_log_prob, critic_outputs = self.pick_action_and_get_critic_values(self.local_model, self.state, epsilon_exploration)
                self.conduct_action(action)
                self.store_state(self.state)
                self.store_action(self.action)
                self.store_reward(self.reward)
                self.store_log_probabilities(action_log_prob)
                self.store_critic_outputs(critic_outputs)
                self.state = self.next_state

            total_loss = self.calculate_total_loss()

        
            self.calculate_and_store_gradients_in_queue(total_loss)
            self.episode_number += 1
            with self.counter.get_lock():
                self.counter.value += 1
                self.results_queue.put(np.sum(self.episode_rewards))

        

    def reset_game(self):
        """ Extends the Base_Agent reset_game to include some new arrays """
        Base_Agent.reset_game(self)
        self.episode_log_action_probabilities = []
        self.critic_outputs = []


    """ Policy, Actions and Exploration """
    def calculate_new_exploration(self):
        """Calculates the new exploration parameter epsilon. It picks a random point within 3X above and below the
        current epsilon"""
        with self.counter.get_lock():
            epsilon = 1.0 / (1.0 + (self.counter.value / self.epsilon_decay_denominator))
        epsilon = max(0.0, random.uniform(epsilon / self.exploration_worker_difference, epsilon * self.exploration_worker_difference))
        return epsilon

    def pick_action_and_get_critic_values(self, policy, state, epsilon_exploration=None):
        """Picks an action using the policy"""
        smoothing = 0.001
        state = torch.from_numpy(state).float().unsqueeze(0)
        model_output = policy.forward(state,None)
        actor_output = model_output[:, list(range(self.action_size))] #we only use first set of columns to decide action, last column is state-value
        critic_output = model_output[:, -1]

        if(self.action_mask_required == True): #todo can't use the forward for this mask cause... critic_output
            mask = self.get_action_mask()
            unormed_action_values =  actor_output.mul(mask)
            actor_output =  unormed_action_values/unormed_action_values.sum()
        else:
            mask = None
        
        action_distribution = create_actor_distribution(self.action_types, actor_output, self.action_size)
        action = action_distribution.sample().cpu().numpy()
        if self.action_types == "CONTINUOUS": action += self.noise.sample()
        if self.action_types == "DISCRETE":
            if random.random() <= epsilon_exploration:
                idx_mask = torch.where(mask == 1)[0]
                a = np.random.randint(0, len(idx_mask))
                action = idx_mask[a].item()
                #action = random.randint(0, self.action_size - 1)
            else:
                action = action[0]
        action_log_prob = self.calculate_log_action_probability(action, action_distribution)
        return action, action_log_prob, critic_output

    def calculate_log_action_probability(self, actions, action_distribution):
        """Calculates the log probability of the chosen action"""
        policy_distribution_log_prob = action_distribution.log_prob(torch.Tensor([actions]))
        return policy_distribution_log_prob


    """  Methods to Calculate Loss """
    def calculate_total_loss(self):
        """Calculates the actor loss + critic loss"""
        discounted_returns = self.calculate_discounted_returns()
        if self.normalise_rewards:
            discounted_returns = self.normalise_discounted_returns(discounted_returns)
        critic_loss, advantages = self.calculate_critic_loss_and_advantages(discounted_returns)
        actor_loss = self.calculate_actor_loss(advantages)
        total_loss = actor_loss + critic_loss
        return total_loss

    def calculate_discounted_returns(self):
        """Calculates the cumulative discounted return for an episode which we will then use in a learning iteration"""
        discounted_returns = []
        discounted_reward = 0
        for ix in range(len(self.episode_states)):
            discounted_reward = self.episode_rewards[-(ix + 1)] + self.discount_rate*discounted_reward
            discounted_returns.insert(0,discounted_reward)
        return discounted_returns

    def normalise_discounted_returns(self, discounted_returns):
        """Normalises the discounted returns by dividing by mean and std of returns that episode"""
        mean = np.mean(discounted_returns)
        std = np.std(discounted_returns)
        discounted_returns -= mean
        discounted_returns /= (std + 1e-5)
        return discounted_returns

    def calculate_critic_loss_and_advantages(self, all_discounted_returns):
        """Calculates the critic's loss and the advantages"""
        critic_values = torch.cat(self.critic_outputs)
        advantages = torch.Tensor(all_discounted_returns) - critic_values
        advantages = advantages.detach()
        critic_loss =  (torch.Tensor(all_discounted_returns) - critic_values)**2
        critic_loss = critic_loss.mean()
        
        return critic_loss, advantages

    def calculate_actor_loss(self, advantages):
        """Calculates the loss for the actor"""
        action_log_probabilities_for_all_episodes = torch.cat(self.episode_log_action_probabilities)
        actor_loss = -1.0 * action_log_probabilities_for_all_episodes * advantages
        actor_loss = actor_loss.mean()
        return actor_loss


    """ Method to handle Gradients """
    def calculate_and_store_gradients_in_queue(self, total_loss):
        """Puts gradients in a queue for the optimisation process to use to update the shared model"""
        self.local_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), self.gradient_clipping_norm)
        gradients = [param.grad.clone() for param in self.local_model.parameters()]
        self.gradient_updates_queue.put(gradients)


    """ Trivial Storage """
    def store_critic_outputs(self,critic_outputs):
        """Stores the critic outputs"""
        self.critic_outputs.append(critic_outputs)

    def store_log_probabilities(self, log_probabilities):
        """Stores the log probabilities of picked actions to be used for learning later"""
        self.episode_log_action_probabilities.append(log_probabilities)

    def store_action(self, action):
        ##TODO change this name cause REinforce has the same for different thing##
        """Stores the action picked"""
        self.episode_actions.append(self.action)

    def store_reward(self,reward):
        """Stores the reward picked"""
        self.episode_rewards.append(reward)

    def store_state(self,state):
        """Stores the state picked"""
        self.episode_states.append(state)


    """ Other """
    def set_seeds(self, worker_num):
        """Sets random seeds for this worker"""
        torch.manual_seed(self.config.seed + worker_num)
        self.environment.seed(self.config.seed + worker_num)
