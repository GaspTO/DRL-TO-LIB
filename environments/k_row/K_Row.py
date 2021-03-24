from logging import currentframe
import sys
import copy
from gym.core import ObservationWrapper
import itertools
from six import StringIO
import numpy as np
import gym
from gym import spaces
import torch
sys.path.append("/home/nizzel/Desktop/Tiago/Computer_Science/Tese/DRL-TO-LIB")
from environments.environment_utils import Players, IN_GAME, TERMINAL


'''
-BOARDS ARE NUMPY ARRAY: WITH TWO ARRAYS INSIDE. EACH WITH THE BOARD SHAPE.
-THE FIRST ARRAY HAS 1 IN THE PLACES FOR THE PIECES OF THE CURRENT PLAYER
-THE SECOND ARRAY HAS 1 IN THE PLACES FOR THE PIECES OF THE ADVERSARY PLAYER

-OCASSIONALY, IT WILL MAKE SENSE TO LOOK AT BOARDS FROM A ONE ARRAY PRESPECTIVE
-IN WHICH CASE, 1 WILL BE THE FIRST_PLAYER PIECE AND -1 FOR THE SECOND
'''

FIRST_PLAYER = Players.get_player(1)
SECOND_PLAYER = Players.get_player(2)
TIE_PLAYER = Players.get_tie_player()

'''
TWO BOARD VIEW
'''
EMPTY_PIECE = 0
OCCUPIED_PIECE = 1

'''
OCASIONAL ONE BOARD VIEW
'''
FIRST_PLAYER_BOARD_PIECE = 1
SECOND_PLAYER_BOARD_PIECE = -1

def get_player_piece(player):
    if player == FIRST_PLAYER:
        return FIRST_PLAYER_BOARD_PIECE
    if player == SECOND_PLAYER:
        return SECOND_PLAYER_BOARD_PIECE
    raise ValueError("Invalid player")

def get_opposite_player(player):
    if player == FIRST_PLAYER:
        return SECOND_PLAYER
    if player == SECOND_PLAYER:
        return FIRST_PLAYER
    raise ValueError("Invalid player")

def one_board_view_to_two_board_view(board: np.array, current_player):
    current_piece = get_player_piece(current_player)
    adversary_piece = get_player_piece(get_opposite_player(current_player))
    current_board = np.where(board==current_piece,1,0)
    opposite_board = np.where(board==adversary_piece,1,0)
    return np.array([current_board,opposite_board])

def get_current_player(board_planes: np.array):
    current_board_flat = board_planes[0].flatten()
    adversary_board_flat = board_planes[1].flatten()
    curr = 0
    adv = 0
    for e in current_board_flat:
        if e == OCCUPIED_PIECE:
            curr += 1
    for e in adversary_board_flat:
        if e == OCCUPIED_PIECE:
            adv += 1
    if curr == adv:
        return FIRST_PLAYER
    elif curr + 1 == adv:
        return SECOND_PLAYER
    else:
        raise ValueError("Problem")
        



class K_Row_State():
    def __init__(self,board_planes,current_player,target_length):
        self.board_planes = board_planes
        self.current_player = current_player
        self.target_length = target_length
        self._winner = None
        self._status = None

    def get_current_player(self):
        return self.current_player

    def get_winner(self):
        self.set_status()
        return self._winner

    def get_status(self):
        self.set_status()
        return self._status

    def is_terminal(self):
        self.set_status()
        return TERMINAL == self._status

    def is_current_player_winner(self):
        self.set_status()
        return self._winner == self.current_player

    def get_current_board(self,two_planes=True):
        if two_planes == True:
            return self.board_planes
        else:
            current_piece = get_player_piece(self.current_player)
            adversary_piece = get_player_piece(get_opposite_player(self.current_player))
            return self.board_planes[0]*current_piece + self.board_planes[1]*adversary_piece

    def get_legal_actions(self):
        legal_actions = []
        flat_summed_board = (self.board_planes[0] + self.board_planes[1]).flatten()
        return list(np.where(flat_summed_board == EMPTY_PIECE)[0])

    def coord_to_action(self, x, y):
        if(self.board.shape[0]-1 < x or self.board.shape[1]-1 < y or x <0 or y<0):
            raise ValueError("coord_to_actions translating impossible coordinated")
        return x *self.board.shape[1] + y

    def action_to_coord(self, action):
        if(self.board_planes[0].shape[0]*self.board_planes[0].shape[1] - 1 <action or action < 0):
            raise ValueError("action_to_coord translating impossible action")
        return (action // self.board_planes[0].shape[1],action % self.board_planes[0].shape[0])

    def act(self,action):
        x,y = self.action_to_coord(action)
        board = copy.deepcopy(self.board_planes)
        board_current = board[0]
        board_adversary = board[1]
        assert board_current[x,y] == EMPTY_PIECE and board_adversary[x,y] == EMPTY_PIECE
        board_current[x,y] = OCCUPIED_PIECE
        new_board_planes = np.array([board_adversary,board_current])
        return K_Row_State(new_board_planes, get_opposite_player(self.current_player),self.target_length)
        
    def render(self):
        adversary_piece = get_player_piece(get_opposite_player(self.current_player))
        current_piece = get_player_piece(self.current_player)
        string = str(self.board_planes[0]*current_piece + self.board_planes[1]*adversary_piece)
        return string

    def get_mask(self):
        a = (self.board_planes[0] + self.board_planes[1])
        return np.where(a==OCCUPIED_PIECE,0,1).flatten()

    def set_status(self):
        ''' set self._winner and self._status'''
        if self._winner != None or self._status != None:
            assert self._status != None
            return
        
        def meets_objective(line: np.array):
            assert len(line.shape) == 1
            if line.shape[0] < self.target_length:
                return False
            how_many_in_a_row = 0
            for piece in line:
                how_many_in_a_row = how_many_in_a_row + 1 if piece == OCCUPIED_PIECE else 0
                if how_many_in_a_row == self.target_length:
                    return True
            return False

        current_player_board = self.board_planes[0]
        adversary_player_board = self.board_planes[1]
        for player, plane in [(self.current_player,current_player_board),(get_opposite_player(self.current_player),adversary_player_board)]: 
            for horizontal_line in plane:
                if meets_objective(horizontal_line) == True:
                    self._winner, self._status = player, TERMINAL
                    return
            for vertical_line in plane.transpose():
                if meets_objective(vertical_line) == True:
                    self._winner, self._status = player, TERMINAL
                    return
            for diagonal_number in range(-plane.shape[0]+1,plane.shape[1]):
                if meets_objective(plane.diagonal(diagonal_number)) == True:
                    self._winner, self._status = player, TERMINAL
                    return
                if meets_objective(np.flipud(plane).diagonal(diagonal_number)) == True:
                    self._winner, self._status = player, TERMINAL
                    return 
        assert self._winner == None
        for e in (self.board_planes[0] + self.board_planes[1]).flatten():
            if e == EMPTY_PIECE:
                self._status = IN_GAME
                return
        self._winner, self._status = TIE_PLAYER, TERMINAL
                  
    @staticmethod
    def get_initial_state(board_shape,target_length):
        board_planes = np.array([np.zeros(board_shape),np.zeros(board_shape)])
        return K_Row_State(board_planes,FIRST_PLAYER,target_length)


class K_Row_Env(gym.Env): 
    def __init__(self,board_shape = 3, target_length = 3,inner_state=None):
        if inner_state == None:
            self.target_length = target_length
            if isinstance(board_shape, int):
                self.board_shape = (board_shape, board_shape)
            assert len(self.board_shape) == 2  # invalid board shape
            self.reset()
        else:
            self.k_row_state = inner_state
            self.target_length = inner_state.target_length
            self.board_shape = inner_state.board_planes[0].shape
            self.done = inner_state.is_terminal()
    
    def step(self, action):
        if self.done == True:   raise ValueError("Playing after game is done")
        self.k_row_state = self.k_row_state.act(action)
        if self.k_row_state.is_terminal():
            self.done = True
            reward = 1.0 if self.k_row_state.get_winner() != TIE_PLAYER else 0.0      #it won - the reward is always on the prespective of who won
        else:
            reward = 0.0
        return self.k_row_state.get_current_board(), reward, self.done, self.get_info()

    def reset(self):
        self.done = False
        self.k_row_state = K_Row_State.get_initial_state(self.board_shape,self.target_length)
        return self.k_row_state.get_current_board(two_planes=True)

    def render(self):
        return self.k_row_state.render()

    def seed(self, seed=None):
        return []

    def close(self):
        return None

    def get_mask(self):
        return self.k_row_state.get_mask()

    def get_legal_actions(self):
        return self.k_row_state.get_legal_actions()

    def is_terminal(self):
        return self.done

    def get_info(self):
        return {"inner_state":self.k_row_state,"target_length":self.target_length,"board_shape":self.board_shape,"done":self.done}

    def get_current_observation(self):
        return self.k_row_state.get_current_board()

    def get_winner(self):
        if self.done is False: raise ValueError("No Winner")
        else: self.k_row_state.get_winner()

    def get_current_player(self):
        return self.k_row_state.get_current_player()

    def get_action_size(self):
        return self.board_shape[0] * self.board_shape[1]

    def get_input_shape(self):
        return self.k_row_state.board_planes.shape
        






'''
a = np.array(
      [[[1., 0., 0., 1.],
        [0., 1., 0., 0.],
        [1., 0., 1., 1.],
        [0., 0., 1., 1.]],

       [[0., 1., 0., 1.],
        [1., 0., 1., 1.],
        [0., 0., 0., 1.],
        [1., 1., 0., 0.]]])



s = K_Row_State(a,get_current_player(a),3)        

r = s.is_terminal()

print(r)
'''

'''
s = K_Row_Env(board_shape = 3, target_length = 3)
s.step(0)
s.step(8)
s.step(2)
s.step(1)
s.step(4)
s.step(6)
s.step(3)
s.step(7)
print(s.render())
'''