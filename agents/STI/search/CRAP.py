import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
sys.path.append("/home/nizzel/Desktop/Tiago/Computer_Science/Tese/DRL-TO-LIB")


from abc import abstractmethod
from random import randint
from typing import AbstractSet
from math import sqrt,log
from agents.STI.Search_Node import Search_Node
from environments.Custom_K_Row import Custom_K_Row
from environments.core.Players import Players, Player, IN_GAME, TERMINAL, TIE_PLAYER_NUMBER




'''
class TSI:
    def __init__(self,root):
        self.root = root
        self.current_node = self.root

    def playout(self,k):
        for _ in range(k):
            self.step()
            self.expand()
        self.backtrack()
'''

        

class mcts:
    def __init__(self,environment):
        self.environment = environment
        self.observation = self.environment.get_current_observation()
        self.root =  Search_Node(self.environment,self.observation,initializer_fn=None)
        self.root.belongs_to_tree = True
        self.current_node = self.root
        print("eval_fn is NONE")


    def playout(self):
        print("__________")
        self.step()
        delta = self.expand() #delta is propagated value in the prespective of the node that was leaf
        self.backtrack(delta)
        print("summarize:" + str(self.root.W/self.root.N))
        print("self.root.W="+str(self.root.W))

    """ step """
    def step(self,eval_fn=None):
        def uct(node):
            log_N_vertex = log(node.get_parent_node().N)
            return -1 * node.W / node.N + sqrt(log_N_vertex / node.N)
        
       
        while self.current_node.is_completely_expanded() and not self.current_node.is_terminal():
            self.current_node = max(self.current_node.get_successors(),key=uct)
            print("Sel:\n" + str(self.current_node.get_parent_action()))


        #! what happens if the I get to a terminal?


    """ expand """
    def expand(self, expansion_fn=None,estimation_fn=None,aggretate_fn=None):
        if not self.current_node.is_terminal():
            #* expand random node
            succ_node = self.current_node.expand_random_successor()
            succ_node.belongs_to_tree = True
            assert succ_node.N == 0
            assert succ_node.W == 0
            succ_node.N = 1

            #* estimation: random rolllout
            rollout_node = succ_node
            while not rollout_node.is_terminal():
                rollout_node = rollout_node.find_random_unexpanded_successor()
        
            
            if rollout_node.get_parent_node().get_current_player() == succ_node.get_current_player(): 
                succ_node.W = rollout_node.get_parent_reward()
            else:
                succ_node.W = -1*rollout_node.get_parent_reward()
            
            if(rollout_node.get_parent_reward() == 1.0):
                print(str(rollout_node.get_parent_node().get_current_player().get_number()) + " ganhou in rollout")
            elif(rollout_node.get_parent_reward() == -1.0):
                print(str(rollout_node.get_current_player().get_number()) + " ganhou in rollout")
            elif rollout_node.get_parent_reward() == 0.0:
                print("EMPATE in rollout")   
            else:
                raise ValueError("What the hell?")         
            return -1*succ_node.W #value to propagate
    
        else:
            if(self.current_node.get_parent_reward() == 1.0):
                print(str(self.current_node.get_parent_node().get_current_player().get_number()) + " ganhou in select")
            elif(self.current_node.get_parent_reward() == -1.0):
                print(str(self.current_node.get_parent_node().get_current_player().get_number()) + " ganhou in select")
            elif self.current_node.get_parent_reward() == 0.0:
                print("EMPATE in select")   
            else:
                raise ValueError("What the hell?")  
            return -1 * self.current_node.get_parent_reward()
            


    """ backward """
    def backtrack(self,delta):
        p = self.current_node.get_current_player()
        self.current_node.W += delta
        self.current_node.N += 1
        while not self.current_node.is_root():
            self.current_node = self.current_node.get_parent_node()
            self.current_node.N += 1
            if self.current_node.get_current_player() == p:
                self.current_node.W += delta
            else:
                self.current_node.W -= delta
            assert self.current_node.parent_reward == 0
            
            
        
        

    


''' MAIN '''
env = Custom_K_Row(board_shape=3, target_length=3)
agent = mcts(env)
for i in range(10000000000):
    agent.playout()
print("END")

    

