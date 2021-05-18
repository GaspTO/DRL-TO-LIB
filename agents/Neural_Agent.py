import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
import numpy as np
import os


class Neural_Agent(nn.Module):
    def __init__(self,device):
        super().__init__()
        self.device = device
        self.observations = None

    def load_observations(self,observations,device=None):
        device = self.device if self.device is not None else device
        assert isinstance(observations,np.ndarray)
        observations = torch.FloatTensor(observations).to(device)
        if self.observations is None or not torch.equal(observations,self.observations):
            self.reset()
            self.observations = observations               
        return self
    
    def is_loaded(self):
        return self.observations != None

    def reset(self):
        self.observations = None

    def get_observations(self):
        return self.observations

    def set_device(self,device):
        self.device = device

    def get_device(self):
        return self.device

    ''' Q(S,A) '''
    def get_q_values(self,mask: np.ndarray):
        raise NotImplementedError

    ''' P(S,A) '''
    def get_policy_values(self,mask: np.ndarray,apply_softmax):
        raise NotImplementedError

    ''' V(S)'''
    def get_state_value(self,mask,apply_softmax):
        raise NotImplementedError
    
    def forward(self,mask,apply_softmax):
        raise NotImplementedError





'''
class Policy_Value_MLP(Neural_Agent):
    def __init__(self,input_size,output_size,hidden_params,n_layers):
        super().__init__()
        previous_layer_params = input_size
        self.main_net = torch.nn.Sequential(nn.Flatten(state_dim=1))
        for i in range(n_layers):
            self.main_net.add_module(nn.Linear(previous_layer_params,hidden_params))
            self.main_net.add(nn.ReLU())
            previous_layer_params = hidden_params
        self.main_net.add(nn.Linear(previous_layer_params,output_size))


    def forward(self, x, mask, apply_softmax):
        self.x1 = x
        self.logits = self.net(self.x1)
        if(mask is not None):
            self.logits = torch.where(mask == 0,torch.tensor(-1e18),self.logits)
        self.output = self.logits if apply_softmax == False else torch.softmax(self.logits,dim=1)
        return self.output
'''

class Policy_Value_MLP(Neural_Agent):
    def __init__(self):
        super().__init__()
        input_size = 18
        action_size = 9
        self.net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(input_size,300),
            nn.ReLU(),
            nn.Linear(300,300),
            nn.ReLU(),
            nn.Linear(300,300))
        
        self.policy_net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(300,300),
            nn.ReLU(),
            nn.Linear(300,action_size),    
        )

        self.value_net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(300,300),
            nn.ReLU(),
            nn.Linear(300,1),   
        )

    """
    def load_state(self,x):
        
        self.main_logits = self.net(self.x)
        self.policy_logits = self.policy_net(self.main_logits)
        self.value_logit = self.value_net(self.main_logits)
        return self

    ''' Q(S,A) '''
    def get_q_values(self, apply_softmax:bool, mask: np.array= None):
        raise NotImplementedError

    ''' P(S,A) '''
    def get_policy_values(self, apply_softmax:bool, mask: np.array= None):
        if mask is not None:
            self.policy_logits = torch.where(mask == 0,torch.tensor(-1e18),self.policy_logits)
        self.policy_output = self.policy_logits if apply_softmax == False else torch.softmax(self.policy_logits,dim=1)
        return self.policy_output
        
    ''' V(S)'''
    def get_state_value(self):
        return self.value_logit
    """



class Parallel_MLP(Neural_Agent):
    def __init__(self,device,input_size=18,action_size=9,hidden_nodes=300):
        super().__init__(device)

        self.pnet = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(input_size,hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes,hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes,hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes,hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes,action_size),    
        )

        self.vnet = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(input_size,hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes,hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes,hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes,hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes,1),   
        )

        self.qnet = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(input_size,hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes,hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes,hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes,hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes,action_size),    
        )

        self.policy_logits = None
        self.value_logit = None
        self.q_logits = None

    def reset(self):
        super().reset()
        self.policy_logits = None
        self.value_logit = None
        self.q_logits = None

    ''' Q(S,A) '''
    def get_q_values(self,mask: np.ndarray = None,retain_results = False):
        #* prepares mask
        if mask is not None:
            assert isinstance(mask,np.ndarray)
            mask = torch.tensor(mask) 

        if self.q_logits is None or retain_results == False:
            self.q_logits = self.qnet(self.observations)
        
        if mask is not None:
            q_logits = torch.where(torch.tensor(mask) == 0,torch.tensor(0),self.q_logits)
        else:
            q_logits = self.q_logits

        return q_logits

    ''' P(S,A) '''
    def get_policy_values(self, apply_softmax:bool, mask: np.ndarray= None,retain_results = False):
        #prepare mask
        assert isinstance(mask,np.ndarray)
        mask = torch.tensor(mask) 

        if self.policy_logits is None or retain_results == False:
            self.policy_logits = self.pnet(self.observations)
            
        if mask is not None:
            policy_logits = torch.where(mask == 0,torch.tensor(-1e18),self.policy_logits)
        else:
            policy_logits = self.policy_logits
        policy_output = policy_logits if apply_softmax == False else torch.softmax(policy_logits,dim=1)
        return policy_output
        
    ''' V(S)'''
    def get_state_value(self, retain_results = False):
        if self.value_logit is None or retain_results == False:
            self.value_logit = self.vnet(self.observations)
        return self.value_logit



class Parallel_Conv(Neural_Agent):
    def __init__(self,input_size=18,action_size=9,hidden_nodes=300):
        super().__init__()

        self.pnet = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(input_size,hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes,hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes,hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes,hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes,action_size),    
        )

        self.vnet = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(input_size,hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes,hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes,hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes,hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes,1),   
        )

        self.qnet = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(input_size,hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes,hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes,hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes,hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes,action_size),    
        )

        self.policy_logits = None
        self.value_logit = None
        self.q_logits = None

    def reset(self):
        super().reset()
        self.policy_logits = None
        self.value_logit = None
        self.q_logits = None

    ''' Q(S,A) '''
    def get_q_values(self,mask: np.ndarray = None,retain_results = False):
        #* prepares mask
        if mask is not None:
            assert isinstance(mask,np.ndarray)
            mask = torch.tensor(mask) 

        if self.q_logits is None or retain_results == False:
            self.q_logits = self.qnet(self.observations)
        
        if mask is not None:
            q_logits = torch.where(torch.tensor(mask) == 0,torch.tensor(0),self.q_logits)
        else:
            q_logits = self.q_logits

        return q_logits

    ''' P(S,A) '''
    def get_policy_values(self, apply_softmax:bool, mask: np.ndarray= None,retain_results = False):
        #prepare mask
        assert isinstance(mask,np.ndarray)
        mask = torch.tensor(mask) 

        if self.policy_logits is None or retain_results == False:
            self.policy_logits = self.pnet(self.observations)
            
        if mask is not None:
            policy_logits = torch.where(mask == 0,torch.tensor(-1e18),self.policy_logits)
        else:
            policy_logits = self.policy_logits
        policy_output = policy_logits if apply_softmax == False else torch.softmax(policy_logits,dim=1)
        return policy_output
        
    ''' V(S)'''
    def get_state_value(self, retain_results = False):
        if self.value_logit is None or retain_results == False:
            self.value_logit = self.vnet(self.observations)
        return self.value_logit
