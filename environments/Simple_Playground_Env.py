from environments.environment_interface import Environment_Interface

class Simple_Playground_Env(Environment_Interface):
    def __init__(self,environment=None):
        self.environment = environment
        self.adversary_agent = None

    def add_agent(self, agent):
        self.set_adversary_agent(agent)

    def set_adversary_agent(self,agent):
        self.adversary_agent = agent

    def get_adversary_agent(self):
        return self.adversary_agent

    def set_environment(self, environment):
        self.environment = environment
    
    def step(self,action,observation=None,info=None,zero_sum = True, all_info = False):
        if info != None: raise ValueError("Simple_Playergroung is still not prepared for info != None ")
        if self.adversary_agent is None: raise ValueError('Need to add an agent to playground')
        if self.environment is None: raise ValueError('Need to set an environment to playground')
        next_state1, reward1, done1, info1 = self.environment.step(action,observation=observation)
        if(not done1):
            action = self.adversary_agent.play(next_state1)
            next_state2, reward2, done2, info2 = self.environment.step(action,observation=observation)
        else:
            next_state2 = next_state1
            reward2 = 0
            done2 = done1
            info2 = info1
            
        total_reward = reward1 + -1*reward2 if zero_sum is True else reward1
        total_info = (info1,info2) if all_info is True else info2

        return next_state2, total_reward, done2, total_info

    ''' just pass them '''
    def reset(self):
        if self.environment is None: raise ValueError('Need to set an environment to playground')
        return self.environment.reset()

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

    

    



    

