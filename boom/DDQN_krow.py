from agents.DQN_agents.DDQN import DDQN, Config_DDQN
from algorithms.Search import MCTS_Search, K_Row_MCTSNode




class Config_DDQN_krow(Config_DDQN):
    def __init__(self,config=None):
        Config_DDQN.__init__(self,config)


class DDQN_krow(DDQN):
    """A double DQN agent"""
    agent_name = "DDQN"

    def __init__(self, config):
        DDQN.__init__(self, config)
        if(self.get_environment_title() != 'K_Row'): raise ValueError("This algorithm only supports the K_ROW game")


    def conduct_action(self):
        action = self.pick_action()
        next_state, reward1, done, _ = self.environment.step(action)
        reward2 = 0
        if done == False:
            search = MCTS_Search(K_Row_MCTSNode(self.environment.state))
            search.run_n_playouts(200)
            action = search.play_action()
            next_state, reward2, done, _ = self.environment.step(action)
        self.action = action
        self.reward = reward1 + reward2
        self.next_state = next_state
        self.done = done
        self.save_transition(transition=(self.state,self.action,self.reward,self.next_state,self.done))
        self.state = self.next_state #only update state after saving transition
        self.episode_states.append(self.state)
        self.episode_actions.append(action)
        self.episode_rewards.append(self.reward)
        self.total_episode_score_so_far += self.reward
        if self.config.get_clip_rewards(): self.reward =  max(min(self.reward, 1.0), -1.0)
        if(self.done == True):
            self.logger.info("final_reward: {}".format(self.reward))

