from agents.search_agents.Search_Node import Search_Node


class Best_First_Search_Node(Search_Node):
    def __init__(self,environment_interface,observation,parent_node=None,parent_action=None,parent_reward=0,terminal=None,legal_actions=None):
        super().__init__(environment_interface,observation,parent_node=parent_node,parent_action=parent_action,parent_reward=parent_reward,terminal=terminal,legal_actions=legal_actions)
        self.num_visits = 0
        self.total_value = 0
        self.exploration_bias = 1

    def find_successor_after_action(self,action):
        new_observation, reward , done , new_game_info = self.environment.step(action,observation=self.observation)
        legal_actions = self.environment.get_legal_actions(observation=new_observation)
        return Best_First_Search_Node(self.environment,new_observation, parent_node=self,parent_action=action,parent_reward=reward,terminal=done,legal_actions=legal_actions)

    

