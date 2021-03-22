import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from agents.Base_Agent import Base_Agent, Config_Base_Agent
from agents.policy_gradient_agents.REINFORCE import REINFORCE, Config_Reinforce
from boom.Search import MCTS_Search, MCTS_Search_attempt_muzero, K_Row_MCTSNode



class Config_Reinforce_adv(Config_Reinforce):
    def __init__(self,config=None):
        Config_Reinforce.__init__(self,config)


class REINFORCE_adv(REINFORCE):
    agent_name = "REINFORCEadv_krow_mcts_vs_mcts"
    def __init__(self, config):
        self.test = 6000
        REINFORCE.__init__(self, config)
        if(self.get_environment_title() != 'K_Row'): raise ValueError("This algorithm only supports the K_ROW game")
 
    def reset_game(self):
        super().reset_game()
        self.episode_rewards1 = []
        self.episode_rewards2 = []
        self.episode_action_log_probabilities1 = []
        self.episode_action_log_probabilities2 = []
        self.episode_step_number = 0

        self.episode_inputs = []
        self.to_learn = True
        self.aux = []
        self.episode_action_probabilities1 = []
        self.episode_action_probabilities2 = []

    def step(self):      
        self.reward1 = None
        self.reward2 = None
        while not self.done:
            '''
            if(self.episode_number > self.test and self.episode_number<(self.test + 100)):
                self.conduct_action(one=self.rl,two=self.mcts,itera=25)
            else:
                self.conduct_action(one=self.rl,two=self.rl,itera=25)
            '''
            self.conduct_action(one=self.rl,two=self.mcts,itera=25)

            if self.done and self.to_learn:
                self.learn()
                self.log_updated_probabilities()
            self.episode_step_number += 1
        self.episode_number += 1
        if(self.debug_mode):
            self.log_updated_probabilities()

    def learn(self):
        policy_loss1, policy_loss2 = 0,0
        if len(self.episode_action_log_probabilities1) == 0 and len(self.episode_action_log_probabilities2) == 0: return
        if len(self.episode_action_log_probabilities1) != 0:
            policy_loss1 = self.calculate_policy_loss_on_episode(episode_rewards=self.episode_rewards1,episode_log_probs=self.episode_action_log_probabilities1)
        if len(self.episode_action_log_probabilities2) != 0:
            policy_loss2 = self.calculate_policy_loss_on_episode(episode_rewards=self.episode_rewards2,episode_log_probs=self.episode_action_log_probabilities2)
        policy_loss = policy_loss1 + policy_loss2
        self.take_optimisation_step(self.optimizer,self.policy,policy_loss,self.config.get_gradient_clipping_norm())
        self.log_updated_probabilities()


    def conduct_action(self,one,two,itera = 5):
        self.reward1 = self.conduct_first_action(one,itera)
        if self.reward2 is not None:  self.episode_rewards2.append((self.reward1 + self.reward2)*-1)
        if(not self.done):
            self.reward2 = self.conduct_second_action(two,itera)
            self.episode_rewards1.append(self.reward1 + self.reward2)
            self.episode_rewards.append(self.reward1 + self.reward2)
            self.total_episode_score_so_far += (self.reward1 + self.reward2)
            if(self.done):
                self.episode_rewards2.append(self.reward2*-1)
        else:
            self.episode_rewards1.append(self.reward1)
            self.episode_rewards.append(self.reward1)
            self.total_episode_score_so_far += self.reward1

    def conduct_first_action(self,agent,n):
        self.episode_inputs.append(self.state)
        action1 = agent(1,n)
        next_state1, reward1, done, _ = self.environment.step(action1)
        self.state = next_state1
        self.action = action1
        self.reward = reward1
        self.done = done
        self.episode_actions.append(action1)
        self.episode_states.append(self.environment.state.board)
        return reward1

    def conduct_second_action(self,agent,n):
        action2 = agent(2,n)
        next_state2, reward2, done, _ = self.environment.step(action2)
        self.state = next_state2
        self.action = action2
        self.reward = reward2
        self.done = done
        if self.done: self.episode_states.append(self.environment.state.board)
        return reward2

    def rl(self,player_n,n = None):
        assert player_n == 1 or player_n == 2
        action, logg =  self.pick_action_and_get_log_probabilities()
        if player_n == 1:
            self.episode_action_log_probabilities1.append(logg)
        if player_n == 2:
            self.episode_action_log_probabilities2.append(logg)
        return action

    def mcts(self,player_n, n = 5):
        assert player_n == 1 or player_n == 2
        search = MCTS_Search(K_Row_MCTSNode(self.environment.state))
        search.run_n_playouts(n)
        action = search.play_action()
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
            self.episode_action_log_probabilities1.append(logg)
        if player_n == 2:
            self.episode_action_log_probabilities2.append(logg)
        return action
        
    def log_updated_probabilities(self,print_results=False):
        r = self.reward
        full_text = []
        for s,a,l in zip(self.episode_inputs,self.episode_actions,self.episode_action_log_probabilities1):
            with torch.no_grad():
                state = torch.from_numpy(s).float().unsqueeze(0).to(self.device)
            actor_values = self.policy(state)
            text = """\r D_reward {0: .2f}, action: {1: 2d}, | old_prob: {2: .10f}, new_prob: {3: .10f}"""
            formatted_text = text.format(r,a,math.exp(l),actor_values[0][a].item())
            if(print_results): print(formatted_text)
            full_text.append(formatted_text )
        self.logger.info("Updated probabilities and Loss After update:" + ''.join(full_text))

    def see_state(self):
        return self.policy(torch.from_numpy(self.state).float().unsqueeze(0).to(self.device))
      