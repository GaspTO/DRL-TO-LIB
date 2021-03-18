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


EMPTY = 0
FIRST_PLAYER = 1
SECOND_PLAYER = -1
TIE = 0




class State():
    def __init__(self,board,player,target_length):
        self.board = board
        self.player = player
        self.target_length = target_length
        self.terminal = None #is it terminal?
        self.winner = None 

    def get_two_boards(self):
        ''' [board1, board2] - board1 full of 1s for current player and board2 full of 1s of opposite player '''
        current_player = np.where(self.board==self.player,1,0)
        opposite_player = np.where(self.board==(-1*self.player),1,0)
        return np.array([current_player,opposite_player])

    def is_terminal(self):
        if(self.terminal == None):
            self.get_winner() #sets self.terminal
        return self.terminal

    def is_winner(self):
        if(self.is_terminal == False):
            raise ValueError("Asking For Winner before finishing game")
        if(self.winner == None): #shouldn't be necessary
            self.get_winner()
        return self.winner

    def get_next_board(self, board, action):
        """
        Get the next state.
        
        Parameters
        ----
        state : np.array    board 
        action : int    location and skip indicator
        
        Returns
        ----
        next_state : np.array   next board 
        
        Raise
        ----
        ValueError : location in action is not valid
        """
        x,y = self.action_to_coord(action)
        if self.is_valid(x,y):
            board = copy.deepcopy(board)
        return board

    def is_valid(self,x,y):
        """
        Check whether the action is valid for current state.
        
        Parameters
        ----
        state : np.array    board and player
        x,y: int  coordinates  location and skip
        
        Returns
        ----
        valid : bool
        """
        if not self.is_index(x,y):
            return False
        #x, y = action
        #x,y = action_to_coord(board.shape,x,y)
        return self.board[x, y] == EMPTY

    def get_valid(self):
        """
        Get all legal actions for the current state.

        Returns
        ----
        valid : list     current valid place for the player
        """
        #valid = np.zeros_like(self.board, dtype=np.int8) 
        valid = []
        for x in range(self.board.shape[0]):
            for y in range(self.board.shape[1]):
                action = self.coord_to_action(x, y)
                if self.is_valid(x,y):
                    valid.append(action)
        return valid

    def has_valid(self):
        """
        Check whether there are valid locations for current state.
        
        Returns
        ----
        has_valid : bool
        """
        for x in range(self.board.shape[0]):
            for y in range(self.board.shape[1]):
                if self.is_valid(x,y):
                    return True
        return False

    def is_index(self, x,y):
        """
        Check whether a location is a valid index of the board
        
        Parameters:
        ----
        x: int
        y: int
        
        Returns
        ----
        is_index : bool
        """
        return x in range(self.board.shape[0]) and y in range(self.board.shape[1])

    def extend_board(self):
        """
        Get the rotations of the board.
        
        Parameters:
        ----
        board : np.array, shape (n, n)
        
        Returns
        ----
        boards : np.array, shape (8, n, n)
        """
        board = self.board
        assert board.shape[0] == board.shape[1]
        boards = np.stack([board,
                np.rot90(board), np.rot90(board, k=2), np.rot90(board, k=3),
                np.transpose(board), np.flipud(board),
                np.rot90(np.flipud(board)), np.fliplr(board)])
        return boards

    def strfboard(self,board, render_characters='+ox', end='\n'):
        """
        Format a board as a string
        
        Parameters
        ----
        board : np.array
        render_characters : str
        end : str
        
        Returns
        ----
        s : str
        """
        s = ''
        for x in range(board.shape[0]):
            for y in range(board.shape[1]):
                c = render_characters[board[x][y]]
                s += c
            s += end
        return s[:-len(end)]

    def coord_to_action(self, x, y):
        if(self.board.shape[0]-1 < x or self.board.shape[1]-1 < y or x <0 or y<0):
            raise ValueError("coord_to_actions translating impossible coordinated")
        return x *self.board.shape[1] + y

    def action_to_coord(self, action):
        if(self.board.shape[0]*self.board.shape[1] - 1 <action or action < 0):
            raise ValueError("action_to_coord translating impossible action")
        return (action // self.board.shape[1],action % self.board.shape[0])

    def act(self,action):
    #def get_next_state_and_player(self, state, action):
        """
        Get the next state.
        
        Parameters
        ----
        state : (np.array, int)    board and current player
        action : int    location and skip indicator
        
        Returns
        ----
        next_state : (np.array, int)    next board and next player
        
        Raise
        ----
        ValueError : location in action is not valid
        """
        x,y = self.action_to_coord(action)
        if self.is_valid(x,y):
            board = copy.deepcopy(self.board)
            board[x, y] = self.player
            return State(board, -self.player,self.target_length)
        else:
            raise ValueError("Play Not Valid")

    def render(self,mode,render_characters):
        """
        See gym.Env.render().
        """
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        s = self.strfboard(self.board, render_characters)
        outfile.write(s)
        if mode != 'human':
            return outfile

    def get_mask(self):
        mask = []
        for row in   self.board:
            for element in row:
                if(element != 0):
                    mask.append(0)
                else:
                    mask.append(1)
        return mask

    def get_winner(self):
        """
        Parameters
        ----
        state : (np.array, int)   board and player. player info is not used
        
        Returns
        ----
        winner : None or int
            - None   if the game is not ended and the winner is not determined
            - int    the winner
        """
        board = self.board
        for player in [FIRST_PLAYER, SECOND_PLAYER]:
            for x in range(board.shape[0]):
                for y in range(board.shape[1]):
                    for dx, dy in [(1, -1), (1, 0), (1, 1), (0, 1)]:  # loop on the 8 directions
                        xx, yy = x, y
                        for count in itertools.count():
                            if not self.is_index(xx,yy) or board[xx, yy] != player:
                            #if not is_index(board, (xx, yy)) or board[xx, yy] != player:
                                break
                            xx, yy = xx + dx, yy + dy
                        if count >= self.target_length:
                            self.terminal = True
                            self.winner = player
                            return player
        for player in [FIRST_PLAYER, SECOND_PLAYER]:
            possible_state = State(board, player,self.target_length)
            if possible_state.has_valid():
                self.terminal = False
                self.winner = None
                return None
        self.terminal = True
        self.winner = 0
        return 0
        

class K_RowEnv(gym.Env):
    metadata = {"render.modes": ["ansi", "human"]}
    
    PASS = -1
    RESIGN = -2
    ERROR = -3
    
    def __init__(self, board_shape = 3, target_length = 3,
            illegal_action_mode='error', render_characters='+ox',
            allow_pass=True):
        """
        Greate a board game.
        
        Parameters
        ----
        board_shape: int or tuple    shape of the board
            - int: the same as (int, int)
            - tuple: in the form of (int, int), the two dimension of the board
        illegal_action_mode: str  What to do when the agent makes an illegal place.
            - 'resign' : invalid location equivalent to resign
            - 'pass' : invalid location equivalent to pass
        render_characters: str with length 3. characters used to render ('012', ' ox', etc)
        """

        self.target_length = target_length
        self.allow_pass = allow_pass
        self.env = "K_Row"

        if illegal_action_mode == 'resign':
            self.illegal_equivalent_action = self.RESIGN
        elif illegal_action_mode == 'pass':
            self.illegal_equivalent_action = self.PASS
        elif illegal_action_mode == 'error':
            self.illegal_equivalent_action = self.ERROR
        else:
            raise ValueError()
        
        self.render_characters = {player : render_characters[player] for player \
                in [EMPTY, FIRST_PLAYER, SECOND_PLAYER]}
        
        if isinstance(board_shape, int):
            self.board_shape = (board_shape, board_shape)
        assert len(self.board_shape) == 2  # invalid board shape
        self.board = np.zeros(self.board_shape)
        assert self.board.size > 1  # Invalid board shape
        
        
        observation_spaces = [
                spaces.Box(low=-1, high=1, shape=self.board_shape, dtype=np.int8),
                spaces.Box(low=-1, high=1, shape=(), dtype=np.int8)]
        self.observation_space = spaces.Tuple(observation_spaces)
        #self.action_space = spaces.Box(low=-np.ones((2,)),
        #        high=np.array(board_shape)-1, dtype=np.int8)
        self.action_space = spaces.Discrete(self.board_shape[0] * self.board_shape[1])
        self.reset()
    

    def get_winner(self, state):
        return state.get_winner()

    def seed(self, seed=None):
        return []
    
    def reset(self):
        """
        Reset a new game episode. See gym.Env.reset()
        
        Returns
        ----
        next_state : (np.array, int)    next board and next player
        """
        self.done = False
        #self.board = np.zeros_like(self.board, dtype=np.int8)
        board = np.zeros(self.board_shape)
        player = FIRST_PLAYER
        self.state = State(board,player,self.target_length)
        return self.state.get_two_boards()
    
    '''def is_valid(self,state, x,y):
        board = state[0]
        return is_valid(board,x,y)

    def get_valid(self,state):
        board = state[0]
        return is_valid(board)

    def has_valid(self,state):
        board = state[0]
        return has_valid(board)'''
    '''
    def get_winner(self, state):
        """
        Check whether the game has ended. If so, who is the winner.
        
        Parameters
        ----
        state : (np.array, int)   board and player. only board info is used
        
        Returns
        ----
        winner : None or int
            - None       The game is not ended and the winner is not determined.
            - env.FIRST_PLAYER  The game is ended with the winner FIRST_PLAYER.
            - env.SECOND_PLAYER  The game is ended with the winner SECOND_PLAYER.
            - env.EMPTY  The game is ended tie.
        """
        board = state.board
        for player in [FIRST_PLAYER, SECOND_PLAYER]:
            if state.has_valid():
                return None
        return np.sign(np.nansum(board))
    '''
    def next_step(self, state, action):
        """
        Get the next observation, reward, done, and info.
        
        Parameters
        ----
        state : (np.array, int)    board and current player
        action : int    location
        
        Returns
        ----
        next_state : (np.array, int)    next board and next player
        reward : float               the winner or zeros
        done : bool           whether the game end or not
        info : {'valid' : np.array, int}    a dict shows the valid place for the next player and next player
        """
        if(self.done == True):
            raise ValueError("Playing after game is done")
        #x,y = action_to_coord(self.board.shape,action)
        #if not self.is_valid(state,x,y):
            #action = self.illegal_equivalent_action
        #if action == self.RESIGN:
            #return state, -state[1], True, {}
        #if action == self.ERROR:
            #raise ValueError("Played in Invalid Position")
        #while True:
        state = self.get_next_state_and_player(state, action)
        winner = self.get_winner(state)
        if winner is not None:
            return state, winner, True, {}
        #if self.has_valid(state):
           #break
        action = self.PASS
        #
        return state, 0., False, {"player":state[1]}
    
    def step(self, action):
        """
        See gym.Env.step().
        
        Parameters
        ----
        action : int   location
        
        Returns
        ----
        next_state : (np.array, int)    next board and next player
        reward : float        the winner or zero
        done : bool           whether the game end or not
        info : {}
        """
        if(self.done == True):
            raise ValueError("Playing after game is done")
        #state = (self.board, self.player)
        #next_state, reward, self.done, info = self.next_step(state, action)
        self.state = self.state.act(action)
        winner = self.get_winner(self.state)
        if winner is not None:
            self.done = True
            return self.state.get_two_boards(), winner, self.done, {"player":self.state.player}
        else:
            return self.state.get_two_boards(), 0, self.done, {"player":self.state.player}
    
    def render(self, mode='human'):
        return self.state.render(mode,self.render_characters)

    def get_mask(self):
        return self.state.get_mask()



'''
a = K_RowEnv(board_shape=3, target_length=3)
a.reset()
a.step(3)
print(a.render())
a.step(4)
print(a.render())
a.step(0)
print(a.render())
a.step(2)
print(a.render())
a.step(6)
print(a.render())
#invalid action
#a.step(1)
#print(a.render())
a.reset()
a.step(0)
a.step(4)
a.step(1)
a.step(2)
a.step(5)
a.step(8)
a.step(7)
a.step(3)
x = a.step(6)
print(x)
print(a.render())
print(a.board)
print("done")
'''
