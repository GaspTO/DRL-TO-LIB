from environments.environment_interface import Environment_Interface
from environments.k_row.K_Row import *


class K_Row_Interface(Environment_Interface):
    def __init__(self,board_shape,target_length):
        self.environment = K_Row_Env(board_shape,target_length)
  
    def step(self,action,info=None):
        if info is None: return self.environment.step(action)
        return K_Row_Env(info=info).step(action)
        
    def reset(self):
        return self.environment.reset()

    def render(self,info=None):
        if info is None: return self.environment.render()
        return info["inner_state"].render()

    def close(self):
        return self.environment.close()

    def seed(self,seed_n):
        return self.environment.seed(seed_n)

    def get_name(self) -> str:
        return "K_Row"

    def needs_mask(self) -> bool:
        return True

    def get_mask(self,info=None):
        if info is None: return self.environment.get_mask()
        return info["inner_state"].get_mask()

    def get_current_observation(self,info=None,human=False):
        if info is None: return self.environment.get_current_observation()
        return info["inner_state"].get_current_board()

    def get_legal_actions(self,info=None):
        if info is None: return self.environment.get_legal_actions()
        return info["inner_state"].get_legal_actions()

    def is_terminal(self, info=None) -> bool:
        if info is None: return self.environment.is_terminal()
        return info["inner_state"].is_terminal()

    def get_game_info(self):
        return self.environment.get_info()

    def get_winner(self, info=None):
        if info is None: return self.environment.get_winner()
        return info["inner_state"].get_winner()

    def get_current_player(self,info=None):
        if info is None: return self.environment.get_current_player()
        return info["inner_state"].get_current_player()
        
    def _make_state_a_new(self,observation):
        return K_Row_State(observation,get_current_player(observation),self.environment.target_length)

