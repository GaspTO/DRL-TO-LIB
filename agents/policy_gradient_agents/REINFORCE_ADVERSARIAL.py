from agents.policy_gradient_agents.REINFORCE import Config_Reinforce, REINFORCE
import torch


class Config_Reinforce_adversarial(Config_Reinforce):
    def __init__(self,config=None):
        Config_Reinforce.__init__(self,config)




class REINFORCE_ADVERSARIAL(REINFORCE):
    def __init__(self,config):
        REINFORCE.__init__(self, config)

    def step(self):
        while not self.done:
            self.conduct_action()
            if self.time_to_learn():
                self.set_terminal_reward() #useful for actor-critic
                self.learn()
            self.episode_step_number += 1
            self.state = self.next_state
        self.episode_number += 1
        if(self.debug_mode):
            self.log_updated_probabilities()
        if(self.done == True):
            self.logger.info("final_reward: {}".format(self.reward))


    def conduct_action(self):
        self.agent1()
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