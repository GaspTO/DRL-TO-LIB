from environments.core.Custom_Environment import Custom_Environment
from utilities.logger import logger

class Custom_Simple_Playground(Custom_Environment):
    '''
    #! Right now the Custom_Simple_Playground and the agents added need to have the same
    #! instance of the same environment
    #zero_sum_game: one agent's rewards is their reward minus the other agent's reward
    #all_info: get the info of the agent and the other agent's in a tuple
    #play_first: True if our agent plays first, False if it plays second
    '''
    def __init__(self,environment=None,zero_sum_game=True,all_info=False,play_first=True):
        self.environment = environment
        self.zero_sum_game = zero_sum_game
        self.all_info = all_info
        self.play_first = play_first
        self.adversary_agent = None
        self.play_no = 0
        self.num_of_players = 1
        
    def add_agent(self, agent):
        self.adversary_agent = agent
        self.num_of_players += 1
        assert self.adversary_agent.get_environment() == self.environment
            
    def step(self,action,observation=None):
        assert self.adversary_agent.get_environment() == self.environment
        if self.adversary_agent is None: raise ValueError('Need to add an agent to playground')
        if self.environment is None: raise ValueError('Need to set an environment to playground')
        use_env_internal_state = False if observation is not None else True
        if observation is not None: raise ValueError('Simply Playground isn\'t ready for untaggled steps from the internal step')

        #* AGENT PLAY
        logger.info("agent: " + str(action))
        next_state1, reward1, done1, info1 = self.environment.step(action)

        #* ADVERSARY PLAY
        if(not done1):
            action2 = self.adversary_agent.play()
            logger.info("adversary: " + str(action2))
            next_state2, reward2, done2, info2 = self.environment.step(action2)            
        else:
            next_state2 = next_state1
            reward2 = 0
            done2 = done1
            info2 = None
        
        total_reward = reward1 + -1*reward2 if self.zero_sum_game is True else reward1
        total_info = (info1,info2) if self.all_info is True else info2
        return next_state2, total_reward, done2, total_info


    ''' just pass them '''
    def reset(self):
        if self.environment is None: raise ValueError('Need to set an environment to playground')
        self.environment.reset()
        if self.play_first ==  False:
            action = self.adversary_agent.play()
            next_state, _, _, _ = self.environment.step(action)
            logger.info("adversary: " + str(action))
        return next_state


    def render(self,info=None):
        return self.environment.render(info=info)

    def close(self):
        return self.environment.close()

    def seed(self,seed_n):
        return self.environment.seed(seed_n)

    def get_name(self):
        return self.environment.get_name()

    def needs_mask(self) -> bool:
        return self.environment.needs_mask()

    def get_mask(self):
        return self.environment.get_mask()

    def get_current_observation(self,observation=None,human=False):
        return self.environment.get_current_observation(observation=observation,human=human)

    def get_legal_actions(self,observation=None):
        return self.environment.get_legal_actions(observation=observation)

    def is_terminal(self, observation=None) -> bool:
        return self.environment.is_terminal(observation=observation)

    def get_game_info(self):
        return self.environment.get_game_info()

    def get_winner(self, observation=None):
        return self.environment.get_winner(observation=observation)

    def get_current_player(self,observation=None):
        return self.environment.get_current_player(observation=observation)

    def set_current_state(self,observation):
        return self.environment.set_current_state(observation)

    



    

