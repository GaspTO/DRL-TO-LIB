class Environment_Interface():  
    def step(self,action,info=None):
        raise NotImplementedError
        
    def reset(self):
        raise NotImplementedError

    def render(self,info=None):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def seed(self,seed_n):
        raise NotImplementedError

    def get_name(self) -> str:
        raise NotImplementedError

    def needs_mask(self) -> bool:
        raise NotImplementedError

    def get_mask(self,info=None):
        raise NotImplementedError

    def get_current_observation(self,info=None,human=False):
        raise NotImplementedError

    def get_legal_actions(self,info=None):
        raise NotImplementedError

    def is_terminal(self, info=None) -> bool:
        raise NotImplementedError

    def get_game_info(self):
        raise NotImplementedError

    def get_winner(self, info=None):
        raise NotImplementedError

    def get_current_player(self,info=None):
        raise NotImplementedError
        
    def _make_state_a_new(self,observation):
        raise NotImplementedError

    


    
    

