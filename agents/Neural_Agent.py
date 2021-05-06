import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
import numpy as np

class Neural_Agent(nn.Module):
    def load_state(self,x):
        raise NotImplementedError

    ''' Q(S,A) '''
    def get_q_values(self,mask,apply_softmax):
        raise NotImplementedError

    ''' P(S,A) '''
    def get_policy_values(self,mask,apply_softmax):
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

    def load_state(self,x):
        self.x = x
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




class Double_Policy_Value_MLP(Neural_Agent):
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

    def load_state(self,x):
        self.x = x
        self.policy_logits = self.pnet(self.x)
        self.value_logit = self.vnet(self.x)
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
