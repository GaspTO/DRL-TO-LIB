from environments.core.Custom_Environment import Custom_Environment
from environments.k_row.K_Row import *


class Custom_K_Row(Custom_Environment):
    def __init__(self,board_shape,target_length):
        self.board_shape = board_shape
        self.target_length = target_length
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
        return "K_Row_" + str(self.environment.board_shape[0]) + "x" + str(self.environment.board_shape[1]) + "_" + str(self.environment.target_length) + "row"

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

    def set_state(self,observation):
        self.environment = K_Row_Env(inner_state=self._make_state_a_new(observation))

    def _make_state_a_new(self,observation):
        return K_Row_State(observation,get_current_player(observation),self.environment.target_length)

