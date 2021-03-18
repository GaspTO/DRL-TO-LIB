import numpy as np
import gym
from gym import spaces
from gym import error
from gym.utils import seeding
from six import StringIO
import sys, os
import six

from environments.gym_gomoku.envs.util import gomoku_util
from environments.gym_gomoku.envs.util import make_random_policy
from environments.gym_gomoku.envs.util import make_beginner_policy
from environments.gym_gomoku.envs.util import make_medium_policy
from environments.gym_gomoku.envs.util import make_expert_policy
from environments.gym_gomoku.envs.state import GomokuState
from environments.gym_gomoku.envs.state import DiscreteWrapper
from environments.gym_gomoku.envs.state import Board

class GomokuEnv(gym.Env):
    '''
    GomokuEnv environment. Play against a fixed opponent.
    '''
    metadata = {"render.modes": ["human", "ansi"]}
    
    def __init__(self, player_color, opponent, board_size):
        """
        Args:
            player_color: Stone color for the agent. Either 'black' or 'white'
            opponent: Name of the opponent policy, e.g. random, beginner, medium, expert
            board_size: board_size of the board to use
        """
        self.id = "Gomoku"
        self.board_size = board_size
        self.player_color = player_color
        
        self.seed()
        
        # opponent
        self.opponent_policy = None
        self.opponent = opponent
        
        # Observation space on board
        shape = (self.board_size, self.board_size) # board_size * board_size
        self.observation_space = spaces.Box(np.zeros(shape,dtype=np.float32), np.ones(shape,dtype=np.float32),None,np.float32)
        
        # One action for each board position
        self.action_space = DiscreteWrapper(self.board_size**2)
        
        # Keep track of the moves
        self.moves = []
        
        # Empty State
        self.state = None
        
        # reset the board during initialization
        self.reset()
    
    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**32
        return [seed1, seed2]
    
    def reset(self):
        self.state = GomokuState(Board(self.board_size), gomoku_util.BLACK) # Black Plays First
        self.reset_opponent(self.state.board) # (re-initialize) the opponent,
        self.moves = []
        
        # Let the opponent play if it's not the agent's turn, there is no resign in Gomoku
        if self.state.color != self.player_color:
            self.state, _ = self.exec_opponent_play(self.state, None, None)
            opponent_action_coord = self.state.board.last_coord
            self.moves.append(opponent_action_coord)
        
        # We should be back to the agent color
        assert self.state.color == self.player_color
        
        # reset action_space
        self.action_space = DiscreteWrapper(self.board_size**2)
        
        self.done = self.state.board.is_terminal()
        return self.state.board.encode()
    
    def close(self):
        self.opponent_policy = None
        self.state = None
    
    def render(self, mode="human", close=False):
        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write(repr(self.state) + '\n')
        return outfile
    
    def step(self, action):
        '''
        Args: 
            action: int
        Return: 
            observation: board encoding, 
            reward: reward of the game, 
            done: boolean, 
            info: state dict
        Raise:
            Illegal Move action, basically the position on board is not empty
        '''
        assert self.state.color == self.player_color # it's the player's turn
        
        # If already terminal, then don't do anything
        if self.done:
            return self.state.board.encode(), 0., True, {'state': self.state}
        
        # Player play
        prev_state = self.state
        self.state = self.state.act(action)
        self.moves.append(self.state.board.last_coord)
        self.action_space.remove(action) # remove current action from action_space
        
        # Opponent play
        if not self.state.board.is_terminal():
            self.state, opponent_action = self.exec_opponent_play(self.state, prev_state, action)
            self.moves.append(self.state.board.last_coord)
            self.action_space.remove(opponent_action)   # remove opponent action from action_space
            # After opponent play, we should be back to the original color
            assert self.state.color == self.player_color
        
        # Reward: if nonterminal, there is no 5 in a row, then the reward is 0
        if not self.state.board.is_terminal():
            self.done = False
            return self.state.board.encode(), 0., False, {'state': self.state}
        
        # We're in a terminal state. Reward is 1 if won, -1 if lost
        assert self.state.board.is_terminal(), 'The game is terminal'
        self.done = True
        
        # Check Fianl wins
        exist, win_color = gomoku_util.check_five_in_row(self.state.board.board_state) # 'empty', 'black', 'white'
        reward = 0.
        if win_color == "empty": # draw
            reward = 0.
        else:
            player_wins = (self.player_color == win_color) # check if player_color is the win_color
            reward = 1. if player_wins else -1.
        return self.state.board.encode(), reward, True, {'state': self.state}
    
    def exec_opponent_play(self, curr_state, prev_state, prev_action):
        '''There is no resign in gomoku'''
        assert curr_state.color != self.player_color
        opponent_action = self.opponent_policy(curr_state, prev_state, prev_action)
        return curr_state.act(opponent_action), opponent_action
    
    @property
    def _state(self):
        return self.state
    
    @property
    def _moves(self):
        return self.moves
    
    def reset_opponent(self, board):
        if self.opponent == 'random':
            self.opponent_policy = make_random_policy(self.np_random)
        elif self.opponent == 'beginner':
            self.opponent_policy = make_beginner_policy(self.np_random)
        elif self.opponent == 'medium':
            self.opponent_policy = make_medium_policy(self.np_random)
        elif self.opponent == 'expert':
            self.opponent_policy = make_expert_policy(self.np_random)
        else:
            raise error.Error('Unrecognized opponent policy {}'.format(self.opponent))

    def get_mask(self):
        return self.state.board.get_mask()