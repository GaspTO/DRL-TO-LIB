import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))


import copy


'''STATUS'''    
IN_GAME = "STILL_RUNNING"
TERMINAL = "TERMINAL"

''' SPECIAL PLAYERS '''
TIE_PLAYER_NUMBER = -1


class Player:
    def __init__(self,number):
        self.number = number
    
    def get_number(self):
        return self.number

    def __str__(self):
        if self.number == TIE_PLAYER_NUMBER:
            return "PLAYER_TIE"
        return "PLAYER_" + str(self.number)
	

'''
    Static class - the point is for this to be an universal datatype.
    Instead of players being int numbers, they are this class. But, just like the number 1
    is universal, so is the Player 1. A Player 1 is equal in every context
'''
class Players:
    players = {}

    @staticmethod
    def get_player(number):
        if number not in Players.players:
        	Players.players[number] = Player(number)
        return Players.players[number]

    @staticmethod
    def get_tie_player(): #to represent when the winner player is a tie
        if TIE_PLAYER_NUMBER not in Players.players:
            Players.players[TIE_PLAYER_NUMBER] = Player(TIE_PLAYER_NUMBER)
        return Players.players[TIE_PLAYER_NUMBER]














