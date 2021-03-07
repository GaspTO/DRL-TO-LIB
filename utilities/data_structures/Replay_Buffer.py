from collections import namedtuple, deque
import random
import torch
import numpy as np

class Replay_Buffer(object):
    """Replay buffer to store past transitions that the agent can then use for training data"""
    
    def __init__(self, buffer_size, batch_size, seed, device=None):

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.transition = namedtuple("transition", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def add_transition(self, states, actions, rewards, next_states, dones):
        """Adds transition(s) into the replay buffer"""
        if type(dones) == list:
            assert type(dones[0]) != list, "A done shouldn't be a list"
            transitions = [self.transition(state, action, reward, next_state, done)
                           for state, action, reward, next_state, done in
                           zip(states, actions, rewards, next_states, dones)]
            self.memory.extend(transitions)
        else:
            transition = self.transition(states, actions, rewards, next_states, dones)
            self.memory.append(transition)
   
    def sample(self, num_transitions=None, separate_out_data_types=True):
        """Draws a random sample of transition from the replay buffer"""
        transitions = self.pick_transitions(num_transitions)
        if separate_out_data_types:
            states, actions, rewards, next_states, dones = self.separate_out_data_types(transitions)
            return states, actions, rewards, next_states, dones
        else:
            return transitions
            
    def separate_out_data_types(self, transitions):
        """Puts the sampled transition into the correct format for a PyTorch neural network"""
        states = torch.tensor([e.state for e in transitions if e is not None]).float().to(self.device)
        actions = torch.tensor([[e.action] for e in transitions if e is not None]).float().to(self.device)
        rewards = torch.tensor([[e.reward] for e in transitions if e is not None]).float().to(self.device)
        next_states = torch.tensor([e.next_state for e in transitions if e is not None]).float().to(self.device)
        dones = torch.tensor([[int(e.done)] for e in transitions if e is not None]).float().to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def pick_transitions(self, num_transitions=None):
        if num_transitions is not None: batch_size = num_transitions
        else: batch_size = self.batch_size
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)
