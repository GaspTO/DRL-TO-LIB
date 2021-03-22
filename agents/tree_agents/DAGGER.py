from agents.Base_Agent import Base_Agent, Config_Base_Agent
from agents.policy_gradient_agents.REINFORCE import REINFORCE, Config_Reinforce
from agents.tree_agents.Searchfuck import MCTS_Search_attempt_muzero
from agents.tree_agents.Node import K_Row_MCTSNode

import torch.optim as optim
import torch
import numpy as np



class DAGGER(REINFORCE):
    agent_name = "DAGGER"
    def __init__(self, config, expert = None):
        REINFORCE.__init__(self, config)
        self.policy = self.config.architecture()
        self.expert = expert
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)
        self.trajectories = []

    def reset_game(self):
        super().reset_game()
        self.expert_probabilities = []

    def conduct_action(self):
        probs = self.mcts_rl()
        self.expert_probabilities.append(probs)
        super().conduct_action()

    def mcts_rl(self):
        search = MCTS_Search_attempt_muzero(K_Row_MCTSNode(self.environment.environment.state),self.device,debug=True)
        search.run_n_playouts(self.policy,25)
        #action = search.play_action()
        return search.get_probs()

    def learn(self):
        #output = torch.tensor(self.expert.play(np.array(self.episode_states)))
        
        targets = torch.cat(self.episode_action_log_probabilities)
        #output.mul()



