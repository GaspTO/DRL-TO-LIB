import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
sys.path.append("/home/nizzel/Desktop/Tiago/Computer_Science/Tese/DRL-TO-LIB")
from utilities.logger import logger

from agents.tree_agents.Search_Node import Search_Node
from agents.Agent import Agent
from environments.Custom_K_Row import Custom_K_Row
from environments.core.Players import Players, Player, IN_GAME, TERMINAL, TIE_PLAYER_NUMBER
from math import sqrt,log
import random
import numpy as np



'''
AGENT SEARCH ALGORITHM: MONTE CARLO TREE SEARCH
THIS IS THE MAIN AGENT
OTHER MODIFICIATIONS HAVE TO BE OVERRIDED IN INHERITED AGENTS
'''
class MCTS_Search(Agent):
    def __init__(self,environment,n_iterations,exploration_weight = 1.0, debug = False):
        self.environment = environment
        self.n_iterations = n_iterations
        self.exploration_weight = exploration_weight
        self.debug = debug

    '''
    Handle Nodes
        * every time a node is created pass this in the constructor so it knows how to initialize some useful parameters
    '''
    def node_initializer(self,node):
        assert hasattr(node,'num_wins') == False
        node.num_wins = 0
        assert hasattr(node,'num_losses') == False
        node.num_losses = 0
        assert hasattr(node,'num_draws') == False
        node.num_draws = 0
        assert hasattr(node,'num_chosen_by_parent') == False
        node.num_chosen_by_parent = 0
        assert hasattr(node,'belongs_to_tree') == False
        node.belongs_to_tree = False
   
    '''
    AGENT interface and other helpful methods
    '''
    def play(self,observations:np.array = None):
        if observations is None: observation = self.environment.get_current_observation() 
        if len(observations) != 1: raise ValueError("MCTS_Search is not ready for batches bigger than 1")
        observation = observations[0]
        self.root =  Search_Node(self.environment,observation,initializer_fn=self.node_initializer)
        self.root.belongs_to_tree = True
        self.current_node = self.root
        self.run_n_playouts(self.n_iterations)
        probs = self.get_action_probabilities()


         #! DEBUG
        current = self.current_node
        sqrt_N = sqrt(current.num_chosen_by_parent)
        self.stri = "NORMAL MCTS values \n:"
        l = current.get_successors()
        l.sort(key=lambda x: x.get_parent_action())
        for node in current.get_successors():
            #print("parent:"+str(node.get_parent_action()) + "    p:"+ str(node.p)+ "   (loss,draws,games):" + "(" + str(node.num_losses) + "," + str(node.num_draws) + "," + str(node.num_chosen_by_parent) + ")" +"   puct:"+str(puctt(node)) + "    U:"+ str( node.p * sqrt_N /(1 + node.num_chosen_by_parent))     +"     Q:" + str(node.num_losses + 0.5 * node.num_draws/(node.num_chosen_by_parent + 1)))
            self.stri += "\t*************parent:"+str(node.get_parent_action()) + "    p:"+ str(None)+ "   (loss,draws,games):" + "(" + str(node.num_losses) + "," + str(node.num_draws) + "," + str(node.num_chosen_by_parent) + ")" + "\n"
    
        #! END DEBUG

        self.probs = probs
        return np.array([probs.argmax()]), None


    '''
    Return node probabilities
        * return probabilitie vector for all actions of the environment (not just the legal actions)
    '''
    def get_action_probabilities(self) -> np.ndarray:
        #the length of successors is not always the action_size 'cause invalid actions don't become successors
        action_probs = np.zeros(self.environment.get_action_size()) 
        for n in self.root.get_successors():
            action_probs[n.parent_action] = self._score_tactic(n)
        # if a vector is full of zeros 
        if(action_probs.sum() == 0.):
            for n in self.root.get_successors():
                action_probs[n.parent_action] = 1/len(self.root.get_successors()) 
        else:
            action_probs = action_probs/action_probs.sum()
        return action_probs

    def _score_tactic(self,node):
        if node.num_chosen_by_parent == 0:
            return 0.  # avoid unseen moves
        return (node.num_losses + 0.5*node.num_draws) / node.num_chosen_by_parent

    def run_n_playouts(self,n):
        for _ in range(n):
            self._selection_phase()
            self._expansion_phase()
            self._simulation_phase()
            self._backpropagation_phase()
            if self.debug: self._debug_node(self.root)

    '''
    Selection Phase
        * _selection_criteria is default for normal MCTS. 
    '''
    def _selection_phase(self):
        while self.current_node.is_completely_expanded() and not self.current_node.is_terminal():
            self.current_node = self._selection_tactic()
            if self.debug: print("Sel:\n" + str(self.current_node.render()))

    def _selection_tactic(self):
            log_N_vertex = log(self.current_node.num_chosen_by_parent)
            def uct(node):
                assert node.num_chosen_by_parent == node.num_losses + node.num_draws + node.num_wins
                #!opponent_losses = node.num_losses + 0.5 * node.num_draws
                opponent_losses = node.num_losses + -1*node.num_wins
                return opponent_losses / node.num_chosen_by_parent + \
                    self.exploration_weight * sqrt(log_N_vertex / node.num_chosen_by_parent)
            return max(self.current_node.get_successors(), key=uct) 

    '''
    Expansion Phase
        * _expansion_criteria is default
    '''
    def _expansion_phase(self):
        if not self.current_node.is_terminal():
            self.current_node = self._expansion_tactic()
            if self.debug: print("Ex:\n" + str(self.current_node.render()))

    def _expansion_tactic(self):
            node = self.current_node.expand_random_successor()
            node.belongs_to_tree = True
            return node

    '''
    Simulation Phase
        * _fast_generation_policy is default
    '''
    def _simulation_phase(self):
        while not self.current_node.is_terminal():
            self.current_node = self._fast_generation_tactic()
            if self.debug: print("Sim:\n" + str(self.current_node.render()))

    def _fast_generation_tactic(self):
            node = self.current_node.find_random_unexpanded_successor()
            node.belongs_to_tree = False
            return node

    '''
    Backpropagation Phase
        * _backpropagate_update_nodes is default
    '''
    def _backpropagation_phase(self):
        self.last_node = self.current_node
        self.winner = self.current_node.get_winner()
        if self.debug == True:
            if self.winner.get_number() == TIE_PLAYER_NUMBER: print("TIE")
            if self.winner.get_number() == 1: print("ONE")
            if self.winner.get_number() == 2: print("TWO")

        while not self.current_node.is_root():
            self._backpropagate_current_node_tactic()
            self.current_node = self.current_node.get_parent_node()
        self._backpropagate_current_node_tactic()

    def _backpropagate_current_node_tactic(self):
        if self.current_node.belongs_to_tree == False:
            return
        if self.winner == Players.get_tie_player():
            self.current_node.num_draws += 1
        elif self.winner == self.current_node.get_current_player(): 
            self.current_node.num_wins += 1
        else:
            self.current_node.num_losses += 1
        self.current_node.num_chosen_by_parent += 1

    
    '''
    Other Operations
        * _debug_node: meaningless debug function.
    '''
    def _debug_node(self,node):
        for n in node.get_successors():
            print("action:" + str(n.parent_action) + " times_chosen: " + str(n.num_chosen_by_parent) +  " losses: " + str(n.num_losses) + " wins: " + str(n.num_wins))




#!Test:#
'''
env = Custom_K_Row(board_shape=3, target_length=3)
agent = MCTS_Search(env)
agent.run_n_playouts(100000)
v = agent.get_action_probabilities()
print(v.sum())
print(v)
print("END")
'''


