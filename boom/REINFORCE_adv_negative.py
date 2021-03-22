import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from agents.Base_Agent import Base_Agent, Config_Base_Agent
from agents.policy_gradient_agents.REINFORCE import REINFORCE, Config_Reinforce
from boom.Search import MCTS_Search, MCTS_Search_attempt_muzero, K_Row_MCTSNode
from boom.REINFORCE_adv import REINFORCE_adv, Config_Reinforce_adv



class Config_Reinforce_adv_negative(Config_Reinforce_adv):
    def __init__(self,config=None):
        Config_Reinforce_adv.__init__(self,config)


class REINFORCE_adv_negative(REINFORCE_adv):
    agent_name = "REINFORCEadv_krow_mcts_vs_mcts"
    def __init__(self, config):
        REINFORCE_adv.__init__(self, config)
        
 
    def reset_game(self):
        super().reset_game()
        self.episode_action_probabilities1 = []
        self.episode_action_probabilities2 = []


    def learn(self):
        policy_loss1, policy_loss2 = 0,0
        if len(self.episode_action_log_probabilities1) == 0 and len(self.episode_action_log_probabilities2) == 0: return
        if len(self.episode_action_probabilities1) != 0:
            policy_loss1 = self.calculate_policy_loss_on_episode_trick(episode_rewards=self.episode_rewards1, episode_probs=self.episode_action_probabilities1)
        if len(self.episode_action_probabilities2) != 0:
            policy_loss2 = self.calculate_policy_loss_on_episode_trick(episode_rewards=self.episode_rewards1, episode_probs=self.episode_action_probabilities1)
        policy_loss = policy_loss1 + policy_loss2
        self.take_optimisation_step(self.optimizer,self.policy,policy_loss,self.config.get_gradient_clipping_norm())
        self.log_updated_probabilities()



    def rl(self,player_n,n = None):
        assert player_n == 1 or player_n == 2
        action, logg, prob =  self.pick_action_and_get_log_probabilities_2()
        if player_n == 1:
            self.episode_action_probabilities1.append(prob)
        if player_n == 2:
            self.episode_action_probabilities2.append(prob)
        return action


    def mcts_rl(self,player_n,n = 5):
        assert player_n == 1 or player_n == 2
        search = MCTS_Search_attempt_muzero(K_Row_MCTSNode(self.environment.state),self.device)
        search.run_n_playouts(self.policy,5)
        action = search.play_action()

        state = torch.from_numpy(self.state).float().unsqueeze(0).to(self.device)
        action_values = self.policy(state)
        logg =  torch.log(action_values[0][torch.tensor([action])])
        if player_n == 1:
            self.episode_action_probabilities1.append(logg)
        if player_n == 2:
            self.episode_action_probabilities2.append(logg)
        return action
        

    
    def calculate_policy_loss_on_episode_trick(self,alpha=1,episode_rewards=None,episode_probs=None):
        if episode_rewards is None: episode_rewards = self.episode_rewards
        if episode_probs is None: episode_probs = self.episode_action_probabilities

        all_discounted_returns = torch.tensor(self.calculate_discounted_returns(episode_rewards=episode_rewards))

        new_p = torch.cat(episode_probs)
        new_p = torch.where(all_discounted_returns>=0,new_p,1-new_p) #this is the trick
        new_log_p = torch.log(new_p)

        advantages = all_discounted_returns
        advantages = advantages.detach()

        action_log_probabilities_for_all_episodes = new_log_p

        actor_loss_values = -1 * action_log_probabilities_for_all_episodes * advantages
        actor_loss =   actor_loss_values.mean() * alpha
        if(self.debug_mode):
            self.set_debug_variables(actor_loss_values_debug=actor_loss_values,\
                actor_loss_debug=actor_loss)

        return actor_loss


    def pick_action_and_get_log_probabilities_2(self):
        """Picks actions and then calculates the log probabilities of the actions it picked given the policy"""
        state = torch.from_numpy(self.state).float().unsqueeze(0).to(self.device)
        action_values = self.policy(state)
        action_values_copy = action_values.detach()
        if(self.action_mask_required == True): #todo can't use the forward for this mask cause... critic_output
            mask = self.get_action_mask()
            unormed_action_values_copy =  action_values_copy.mul(mask)
            action_values_copy =  unormed_action_values_copy/unormed_action_values_copy.sum()
        action_distribution = Categorical(action_values_copy) # this creates a distribution to sample from
        action = action_distribution.sample()
        if(self.debug_mode): self.logger.info("Q values\n {} -- Action chosen {} Masked_Prob {:.5f} True_Prob {:.5f}".format(action_values, action.item(),action_values_copy[0][action].item(),action_values[0][action].item()))
        else: self.logger.info("Action chosen {} Masked_Prob {:.5f} True_Prob {:.5f}\n".format(action.item(),action_values_copy[0][action].item(),action_values[0][action].item()))
        return action.item(), torch.log(action_values[0][action]), action_values[0][action]

    def log_updated_probabilities(self,print_results=False):
        r = self.reward
        full_text = []
        for s,a,p in zip(self.episode_inputs,self.episode_actions,self.episode_action_probabilities1):
            with torch.no_grad():
                state = torch.from_numpy(s).float().unsqueeze(0).to(self.device)
            actor_values = self.policy(state)
            text = """\r D_reward {0: .2f}, action: {1: 2d}, | old_prob: {2: .10f}, new_prob: {3: .10f}"""
            formatted_text = text.format(r,a,p.item(),actor_values[0][a].item())
            if(print_results): print(formatted_text)
            full_text.append(formatted_text )
        self.logger.info("Updated probabilities and Loss After update:" + ''.join(full_text))

    def see_state(self):
        return self.policy(torch.from_numpy(self.state).float().unsqueeze(0).to(self.device))
      