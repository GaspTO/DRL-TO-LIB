from agents.policy_gradient_agents.REINFORCE import REINFORCE, Config_Reinforce
import torch
import torch.nn as nn
from logic.gomoku_search import Gomoku_Tree, Node
from torch.distributions import Categorical


class Config_Reinforce_Tree(Config_Reinforce):
    def __init__(self,config=None):
        Config_Reinforce.__init__(self,config)


class Policy_Reinforce_Tree(nn.Module):
    def __init__(self):
        super().__init__()
        self.one = nn.Sequential(
            nn.Linear(81, 30),
            nn.Sigmoid(),
            nn.Linear(30,30),
            nn.Sigmoid(),
            nn.Linear(30,30),
            nn.Sigmoid()
        )
        self.actions =  nn.Sequential(
            nn.Linear(30,81),
            #todo 
        ) 
        self.critic = nn.Sequential(
            nn.Linear(30,1)
        )

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        self.x1 = x.view(x.size(0),-1)
        self.x2 = self.one(self.x1)


        self.actions1 = self.actions(self.x2)
        self.actions2 = torch.softmax(self.actions1,dim=1)
        #self.actions3 = (self.actions2 + smoothing) 
        #self.actions4 = self.actions3/self.actions3.sum()

 
        self.x2.retain_grad()
        #self.actions1.retain_grad()
        self.actions2.retain_grad()
        #self.actions3.retain_grad()
        #self.actions4.retain_grad()
        
        #critic = self.critic(self.x2)
        #return self.actions2,critic
        return self.actions2

    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters(), lr=2e-13)
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        return optimizer
    


class REINFORCE_Tree(REINFORCE):
    def pick_action_and_get_log_probabilities(self):
        """Picks actions and then calculates the log probabilities of the actions it picked given the policy"""
        state = torch.from_numpy(self.get_state()).float().unsqueeze(0).to(self.device)
        action_values = self.policy(state,self.get_action_mask().unsqueeze(0))
        action_values_copy = action_values.detach()
        
        if(self.action_mask_required == True): #todo can't use the forward for this mask cause... critic_output
            mask = self.get_action_mask()
            unormed_action_values_copy =  action_values_copy.mul(mask)
            action_values_copy =  unormed_action_values_copy/unormed_action_values_copy.sum()
        
        action_distribution = Categorical(action_values_copy) # this creates a distribution to sample from
        
        search_result = Gomoku_Tree(self.environment.state).BFS(1)
        if(search_result is not None):
            action = torch.tensor([search_result])
        else:
            action = action_distribution.sample()

        if(self.get_action_mask()[action]==0):
            print("fuck")
        
        if(self.debug_mode): self.logger.info("Q values\n {} -- Action chosen {} Masked_Prob {:.5f} True_Prob {:.5f}".format(action_values, action.item(),action_values_copy[0][action].item(),action_values[0][action].item()))
        else: self.logger.info("Action chosen {} Masked_Prob {:.5f} True_Prob {:.5f}".format(action.item(),action_values_copy[0][action].item(),action_values[0][action].item()))
        return action.item(), torch.log(action_values[0][action])
