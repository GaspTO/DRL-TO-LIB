from agents.policy_gradient_agents.REINFORCE import Config_Reinforce, REINFORCE
import torch

class Config_Reinforce_Baseline(Config_Reinforce):
    def __init__(self,config=None):
        Config_Reinforce.__init__(self,config)
        if(isinstance(config,Config_Reinforce_Baseline)):
            self.critic_architecture = self.get_critic_architecture()
            self.critic_learning_rate = self.get_critic_learning_rate()
        else:
            self.critic_architecture = None
            self.critic_learning_rate = config.get_learning_rate()
    
    def get_critic_architecture(self):
        if(self.critic_architecture == None):
            raise ValueError("Critic Architecture Not Defined")
        return self.critic_architecture

    def get_critic_learning_rate(self):
        return self.critic_learning_rate


    
class REINFORCE_BASELINE(REINFORCE):
    agent_name = "REINFORCE_BASELINE"
    def __init__(self, config):
        REINFORCE.__init__(self, config)
        self.critic = self.config.critic_architecture()
        self.config.critic_learning_rate = self.config.get_critic_learning_rate()

    def get_critic_value(self):
        state = torch.from_numpy(self.state).float().unsqueeze(0).to(self.device)
        critic_value = self.critic(state)
        return critic_value

    def calculate_policy_loss_on_episode(self,alpha=1,episode_rewards=None,episode_log_probs=None,episode_critic_values=None):
        self.actor_loss = super().calculate_policy_loss_on_episode()
        critic_loss_values = (self.advantages)**2
        self.critic_loss = critic_loss_values.mean()
        return self.actor_loss + self.critic_loss  
    


