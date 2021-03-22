class Simple_Playground_Env():
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
    
    def step(self,action,zero_sum = True, all_info = False):
        if self.adversary_agent is None: raise ValueError('Need to add an agent to playground')
        if self.environment is None: raise ValueError('Need to set an environment to playground')
        next_state1, reward1, done1, info1 = self.environment.step(action)
        if(not done1):
            action = self.adversary_agent.play(next_state1)
            next_state2, reward2, done2, info2 = self.environment.step(action)
        else:
            next_state2 = next_state1
            reward2 = 0
            done2 = done1
            info2 = None
            
        total_reward = reward1 + reward2 if zero_sum is True else reward1
        total_info = (info1,info2) if all_info is True else info1

        return next_state2, total_reward, done2, total_info

    def reset(self):
        if self.environment is None: raise ValueError('Need to set an environment to playground')
        return self.environment.reset()

    def get_id(self):
        return str(self.environment.env)

    def seed(self,seed_n):
        return self.environment.seed(seed_n)

    def get_mask(self):
        return self.environment.get_mask()

    



    

