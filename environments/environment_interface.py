
    


class Environment_Interface():
    def step(self,action,observation=None):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def seed(self,seed_n):
        raise NotImplementedError

    def get_name(self) -> str:
        raise NotImplementedError

    def needs_mask(self) -> bool:
        raise NotImplementedError

    def get_mask(self):
        raise NotImplementedError
    
    def get_current_state(self,human=False):
        raise NotImplementedError

    def get_legal_actions(self,observation=None):
        raise NotImplementedError

    def is_terminal(self, observation=None) -> bool:
        raise NotImplementedError

    


    
    

