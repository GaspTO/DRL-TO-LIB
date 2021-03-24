import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
#sys.path.append("/home/nizzel/Desktop/Tiago/Computer_Science/Tese/DRL-TO-LIB")
from agents.Agent import Agent
from environments.k_row_interface import K_Row_Interface
from environments.environment_utils import Players, Player, IN_GAME, TERMINAL
from math import sqrt,log
import random
import numpy as np



''' Node '''
class MCTS_Node():
    def __init__(self,environment_interface,observation,parent_node=None,parent_action=None,terminal=None,legal_actions=None):
        self.environment = environment_interface
        self.observation = observation
        self.parent_node = parent_node
        self.parent_action = parent_action
        self.successors = []
        self.depth = 0 if parent_node is None else self.parent_node.depth + 1
        self.terminal = terminal if terminal != None else self.environment.is_terminal(observation=observation)
        self.all_legal_actions = legal_actions if legal_actions != None else self.environment.get_legal_actions(observation=observation)
        self.non_expanded_legal_actions = legal_actions
        ### aux
        self.num_wins = 0
        self.num_losses = 0
        self.num_draws = 0
        self.num_chosen_by_parent = 0
        self.belongs_to_tree = None
        
    ''' find '''
    def find_successor_after_action(self,action):
        new_observation, _ , done , new_game_info = self.environment.step(action,observation=self.observation)
        legal_actions = self.environment.get_legal_actions(observation=new_observation)
        return MCTS_Node(self.environment,new_observation,parent_node = self,parent_action=action,terminal = done,legal_actions=legal_actions)

    def find_random_unexpanded_successor(self):
        random_idx = random.randint(0,len(self.non_expanded_legal_actions)-1)
        action = self.non_expanded_legal_actions[random_idx]
        return self.find_successor_after_action(action)

    def find_rest_unexpanded_successors(self):
        generation = []
        for action in self.non_expanded_legal_actions:
            generation.append(self.find_successor_after_action(action))
        return generation

    ''' append ''' 
    def append_successors_to_node(self,successors: list):
        for node in successors:
            self.successors.append(node)
            self.get_non_expanded_legal_actions().remove(node.get_parent_action())

    ''' expand '''
    def expand_random_successor(self):
        node = self.find_random_unexpanded_successor()
        self.append_successors_to_node([node])
        return node

    def expand_rest_successors(self):
        nodes = self.find_rest_unexpanded_successors()
        self.append_successors_to_node(nodes)
        return nodes

    """ Getters """
    def is_terminal(self):
        if(self.terminal != None):
            return self.terminal
        else:
            raise ValueError("Impossible to get if is terminal state")

    def is_completely_expanded(self):
        return len(self.get_non_expanded_legal_actions()) == 0

    def get_non_expanded_legal_actions(self):
        if(self.non_expanded_legal_actions is None):
            self.non_expanded_legal_actions = self.get_all_legal_actions()
        return self.non_expanded_legal_actions

    def is_root(self):
        return self.parent_node == None

    def get_depth(self):
        return self.depth

    def get_successors(self):
        return self.successors

    def get_all_legal_actions(self):
        if(self.all_legal_actions != None):
            return self.all_legal_actions
        else:
            raise ValueError("Impossible to get all legal actions")

    def get_parent_node(self):
        return self.parent_node

    def get_parent_action(self):
        return self.parent_action

    def get_winner(self):
        if not self.is_terminal():
            raise ValueError("Node is not Terminal")
        return self.environment.get_winner(observation=self.observation)

    def get_current_player(self):
        return self.environment.get_current_player(observation=self.observation)

    def get_current_observation(self):
        return self.environment.get_current_observation(observation=self.observation)

    def render(self):
        return self.environment.get_current_observation(observation=self.observation)

      
''' Search Algorithm '''
class MCTS_Search():
    def __init__(self,environment,observation = None,n_iterations = None,exploration_weight = 1.0, debug = False):
        if observation is None: observation = environment.get_current_observation() 
        self.environment = environment
        self.root =  MCTS_Node(environment,observation)
        self.root.belongs_to_tree = True
        self.current_node = self.root
        self.exploration_weight = exploration_weight
        self.debug = debug
        self.n_iterations = n_iterations

    def get_play_probabilities(self, n_iterations = 0, debug = False):
        if self.n_iterations != None:
            assert self.n_iterations != None
            n_iterations = self.n_iterations
        self.run_n_playouts(n_iterations)
        def score(node):
            if node.num_chosen_by_parent == 0:
                return float("0.")  # avoid unseen moves
            return (node.num_losses + 0.5*node.num_draws) / node.num_chosen_by_parent 
        action_probs = np.zeros(self.environment.get_action_size()) #the len(successors) is not always the action_size
        for n in self.root.get_successors():
            action_probs[n.parent_action] = score(n)
        if(action_probs.sum() == 0.):
            size = len(self.root.get_successors())
            for n in self.root.get_successors():
                action_probs[n.parent_action] = 1/size 
        else:
            action_probs = action_probs/action_probs.sum()
        if(np.isnan(action_probs).any()):
            print("crap")
        return action_probs

    def play_action(self, n_iterations = 0, debug = False):
        if self.n_iterations != None:
            assert self.n_iterations != None
            n_iterations = self.n_iterations
        self.run_n_playouts(n_iterations)
        def score(node):
            if node.num_chosen_by_parent == 0:
                return float("-inf")  # avoid unseen moves
            return (node.num_losses + 0.5*node.num_draws) / node.num_chosen_by_parent  # choose the board that made the opponent lose the most
        best_action =  max(self.root.get_successors(), key=score).parent_action

        if(self.debug == True or debug == True):
            for n in self.root.get_successors():
                print("action:" + str(n.parent_action) + " score:" + str(score(n)))
            print("best action chosen:" +  str(best_action))
        return best_action

    def run_n_playouts(self,iterations):
        for n in range(iterations):
            self.selection_phase()
            self.expansion_phase()
            self.simulation_phase()
            self.backpropagation_phase()
            if self.debug: self.debug_node(self.root)


    ''' Selection Phase '''
    def selection_phase(self):
        while self.current_node.is_completely_expanded() and not self.current_node.is_terminal():
            self.current_node = self.selection_criteria()
            if self.debug: print("Sel:\n" + str(self.current_node.render()) )

    def selection_criteria(self):
            log_N_vertex = log(self.current_node.num_chosen_by_parent)
            def uct(node):
                assert node.num_chosen_by_parent == node.num_losses + node.num_draws + node.num_wins
                opponent_losses = node.num_losses + 0.5 * node.num_draws
                return opponent_losses / node.num_chosen_by_parent + \
                    self.exploration_weight * sqrt(log_N_vertex / node.num_chosen_by_parent)
            return max(self.current_node.get_successors(), key=uct) 

    ''' Expansion Phase '''
    def expansion_phase(self):
        if not self.current_node.is_terminal():
            self.current_node = self.expansion_criteria()
            if self.debug: print("Ex:\n" + str(self.current_node.render()) )

    def expansion_criteria(self):
            node = self.current_node.expand_random_successor()
            node.belongs_to_tree = True
            return node

    ''' Simulation Phase '''
    def simulation_phase(self):
        while not self.current_node.is_terminal():
            self.current_node = self.fast_generation_policy()
            if self.debug: print("Sim:\n" + str(self.current_node.render()) )

    def fast_generation_policy(self):
            node = self.current_node.find_random_unexpanded_successor()
            node.belongs_to_tree = False
            return node

    ''' Backpropagation Phase '''
    def backpropagation_phase(self):
        self.last_node = self.current_node
        self.winner = self.current_node.get_winner()
        if self.debug == True:
            if self.winner.get_number() == -1: 
                print("TIE")
            if self.winner.get_number() == 1: print("ONE")
            if self.winner.get_number() == 2: print("TWO")

        while not self.current_node.is_root():
            self.backpropagate_update_nodes()
            self.current_node = self.current_node.get_parent_node()
        self.backpropagate_update_nodes()

    def backpropagate_update_nodes(self):
        if self.current_node.belongs_to_tree == False:
            return
        if self.winner == Players.get_tie_player():
            self.current_node.num_draws += 1
        elif self.winner == self.current_node.get_current_player(): 
            self.current_node.num_wins += 1
        else:
            self.current_node.num_losses += 1
        self.current_node.num_chosen_by_parent += 1

    def debug_node(self,node):
        for n in node.get_successors():
            print("action:" + str(n.parent_action) + " times_chosen: " + str(n.num_chosen_by_parent) +  " losses: " + str(n.num_losses) + " wins: " + str(n.num_wins))



class MCTS_Agent(Agent):
    def __init__(self,environment,n_iterations,exploration_weight=1.0):
        super().__init__(environment)
        self.n_iterations = n_iterations
        self.exploration_weight = exploration_weight

    def play(self,observation=None):
        if observation is None: observation = self.environment.get_current_observation()
        search = MCTS_Search(self.environment, observation = observation, n_iterations = self.n_iterations,exploration_weight=self.exploration_weight)
        action = search.play_action()
        return action

    def get_play_probabilities(self,observation=None):
        if observation is None: observation = self.environment.get_current_observation()
        search = MCTS_Search(self.environment, observation = observation, n_iterations = self.n_iterations,exploration_weight=self.exploration_weight)
        probabilities = search.get_play_probabilities()
        return probabilities

''' test search '''
'''
env = K_Row_Interface(board_shape=4, target_length=3)
#search = MCTS_Search(env)
#search.run_n_playouts(1000000)
#p = search.play_action(debug=True)
'''

''' test agent '''
'''
mcts_search = MCTS_Agent(env,250000)
a = mcts_search.play(env.get_game_info())
print(a)
'''



