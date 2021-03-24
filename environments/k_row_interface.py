from environments.environment_interface import Environment_Interface
from environments.k_row.K_Row import *


class K_Row_Interface(Environment_Interface):
    def __init__(self,board_shape,target_length):
        self.environment = K_Row_Env(board_shape,target_length)
  
    def step(self,action, observation=None):
        if observation is None: return self.environment.step(action)
        return K_Row_Env(inner_state=self._make_state_a_new(observation)).step(action)
        
    def reset(self):
        return self.environment.reset()

    def render(self,observation=None):
        if observation is None: return self.environment.render()
        return self._make_state_a_new(observation).render()

    def close(self):
        return self.environment.close()

    def seed(self,seed_n):
        return self.environment.seed(seed_n)

    def get_name(self) -> str:
        return "K_Row"

    def needs_mask(self) -> bool:
        return True

    def get_mask(self,observation=None):
        if observation is None: return self.environment.get_mask()
        return self._make_state_a_new(observation).get_mask()

    def get_current_observation(self,observation=None,human=False):
        if observation is None: return self.environment.get_current_observation()
        return self._make_state_a_new(observation).get_current_board()

    def get_legal_actions(self,observation=None):
        if observation is None: return self.environment.get_legal_actions()
        return self._make_state_a_new(observation).get_legal_actions()

    def is_terminal(self, observation=None) -> bool:
        if observation is None: return self.environment.is_terminal()
        return self._make_state_a_new(observation).is_terminal()

    def get_game_info(self):
        return self.environment.get_info()

    def get_winner(self, observation=None):
        if observation is None: return self.environment.get_winner()
        return self._make_state_a_new(observation).get_winner()

    def get_current_player(self,observation=None):
        if observation is None: return self.environment.get_current_player()
        return self._make_state_a_new(observation).get_current_player()

    def get_action_size(self):
        return self.environment.get_action_size()

    def get_input_shape(self):
        return self.environment.get_input_shape()
        
    def _make_state_a_new(self,observation):
        return K_Row_State(observation,get_current_player(observation),self.environment.target_length)

