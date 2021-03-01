from agents.policy_gradient_agents.REINFORCE import REINFORCE
from torch.distributions import Categorical
import torch

class Logic_Loss_Reinforce(REINFORCE):
    def __init__(self,config):
        REINFORCE.__init__(self,config)

    def pick_action_and_get_log_probabilities(self):
        """Picks actions and then calculates the log probabilities of the actions it picked given the policy"""
        # PyTorch only accepts mini-batches and not individual observations so we have to add
        # a "fake" dimension to our observation using unsqueeze
        state = torch.from_numpy(self.state).float().unsqueeze(0).to(self.device)
        action_values = self.policy.forward(state,self.get_action_mask()).cpu() 
        action_values_copy = action_values.detach()
        action_distribution = Categorical(action_values_copy) # this creates a distribution to sample from
        action = action_distribution.sample()
        self.calculate_logic_loss(action_values)
        return action.item(), torch.log(action_values[0][action])

    def calculate_policy_loss_on_episode(self, total_discounted_reward):
        """Calculates the loss from an episode"""
        policy_loss = []
        logic_loss = []
        for log_prob, log_loss in zip(self.episode_log_probabilities,self.episode_logic_loss):
            policy_loss.append(-log_prob * total_discounted_reward)
            policy_loss.append(log_loss)

            #logic_loss.append(-torch.min(torch.tensor([1.]),torch.tensor([1.])-+torch.log(log_prob))
        policy_loss = torch.cat(policy_loss).sum() # We need to add up the losses across the mini-batch to get 1 overall loss
        
        return policy_loss

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        super().reset_game()
        #todo only black now
        self.episode_logic_loss = []

    def store_critical_actions(self,x):
        """Stores the reward picked"""
        self.episode_critical_actions.append(x)

    def pick_and_conduct_action_and_save_log_probabilities(self):
        
        """Picks and then conducts actions. Then saves the log probabilities of the actions it conducted to be used for
        learning later"""
        action, log_probabilities = self.pick_action_and_get_log_probabilities()
        self.store_log_probabilities(log_probabilities)
        self.store_action(action)
        self.conduct_action(action)

   
    def calculate_logic_loss(self,probabilities):
        critical_actions = self.environment.state.board.get_critical_action()
        loss = torch.tensor([0.])
        for action in critical_actions[0]: #blacks
            loss = loss + (1 - probabilities[0][action].unsqueeze(0))
        for action in critical_actions[1]: #whites
            loss = loss + (1 - probabilities[0][action].unsqueeze(0))
        self.episode_logic_loss.append(loss)



