class Environment_Interface():  
    def step(self,action,observation=None):
        raise NotImplementedError
        
    def reset(self):
        raise NotImplementedError

    def render(self,observation=None):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def seed(self,seed_n):
        raise NotImplementedError

    def get_name(self) -> str:
        raise NotImplementedError

    def needs_mask(self) -> bool:
        raise NotImplementedError

    def get_mask(self,observation=None):
        raise NotImplementedError

    def get_current_observation(self,observation=None,human=False):
        raise NotImplementedError

    def get_legal_actions(self,observation=None):
        raise NotImplementedError

    def is_terminal(self, observation=None) -> bool:
        raise NotImplementedError

    def get_game_info(self):
        raise NotImplementedError

    def get_winner(self, observation=None):
        raise NotImplementedError

    def get_current_player(self,observation=None):
        raise NotImplementedError

    def get_action_size(self):
        raise NotImplementedError
    
    def get_input_shape(self):
        raise NotImplementedError

    


    
    

