import copy

from agents.Base_Agent import Base_Agent
from agents.DQN_agents.DQN import DQN, Config_DQN


class Config_DQN_With_Fixed_Q_Targets(Config_DQN):
    def __init__(self,config=None):
        Config_DQN.__init__(self,config),
        if(isinstance(config,Config_DQN_With_Fixed_Q_Targets)):
            self.tau = config.get_tau()
        else:
            self.tau = 0.01

    def get_tau(self):
        if(self.tau != None):
            return self.tau
        else:
            raise ValueError("Tau Not Defined")
      
class DQN_With_Fixed_Q_Targets(DQN):
    """A DQN agent that uses an older version of the q_network as the target network"""
    agent_name = "DQN with Fixed Q Targets"
    def __init__(self, config):
        DQN.__init__(self, config)
        self.q_network_target = self.create_NN_through_NNbuilder(input_dim=self.input_shape, output_size=self.action_size,smoothing=0.001)
        #self.q_network_target = self.create_NN(input_dim=self.state_size, output_dim=self.action_size)
        Base_Agent.copy_model_over(from_model=self.q_network_local, to_model=self.q_network_target)

    def learn(self, experiences=None):
        """Runs a learning iteration for the Q network"""
        super(DQN_With_Fixed_Q_Targets, self).learn(experiences=experiences)
        self.soft_update_of_target_network(self.q_network_local, self.q_network_target,
                                           self.config.get_tau())  # Update the target network

    def compute_q_values_for_next_states(self, next_states):
        """Computes the q_values for next state we will use to create the loss to train the Q network"""
        Q_targets_next = self.q_network_target(next_states).detach().max(1)[0].unsqueeze(1)
        return Q_targets_next