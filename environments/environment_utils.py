import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))


import copy


'''STATUS'''    
IN_GAME = "STILL_RUNNING"
TERMINAL = "TERMINAL"



class Player:
    def __init__(self,number):
        self.number = number
    
    def get_number(self):
        return self.number

class Players:
    players = {}

    @staticmethod
    def get_player(number):
        if number not in Players.players:
            Players.players[number] = Player(number)
        return Players.players[number]

    @staticmethod
    def get_tie_player(): #to represent when the winner player is a tie
        if -1 not in Players.players:
            Players.players[-1] = Player(-1)
        return Players.players[-1]












