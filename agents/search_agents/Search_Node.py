import random
import copy

from numpy import string_
from environments.core.Players import Players, Player, IN_GAME, TERMINAL
from collections import OrderedDict

'''
    Special Attributes
                    ________
                   |        |   parent_reward of root is always 0.
                   |  NODE  |
                   |________|
                  /          \    
         ________/            \_______      parent_reward  -> reward from the prespective of parent's node
        |        |           |        |     number_of_visits  -> total number of visits
        |  SUCC1 |           |  SUCC2 |     total_reward  -> Sum childs reward + childs rest path reward
        |________|           |________|     temp_rest_path_reward  -> temporary variable to store reward gotten in path
                                  \
                                  /
                                  \
                                (...)    path until terminal node
                                  /
                                  \
                              ________
                             |TERMINAL|
                             |  NODE  |
                             |________|    temp_future_total_reward is always 0

This node is made specifically to search in environments using trees (one parent node)
'''
#! change name to Environment_Node
class Search_Node():
    def __init__(self,environment_interface,observation,parent_node=None,parent_action=None,parent_reward=0,terminal=None,legal_actions=None):
        self.environment = environment_interface
        self.observation = observation
        self.parent_node = parent_node
        self.parent_action = parent_action
        self.parent_reward = parent_reward
        self.depth = 0 if parent_node is None else self.parent_node.depth + 1    
        self.terminal = terminal if terminal != None else self.environment.is_terminal(observation=observation)
        self.successors = dict()
        self.all_legal_actions = list(legal_actions) if legal_actions is not None else list(self.environment.get_legal_actions(observation=observation))
        self.non_expanded_legal_actions = copy.deepcopy(self.all_legal_actions)
        if parent_node is None:
            self.path_reward = 0
        else:
            if self.parent_node.get_player() == self.get_player():
                self.path_reward = self.parent_node.path_reward + parent_reward
            else:
                self.path_reward = -1* self.parent_node.path_reward + -1*parent_reward
        
    
    """ 
    Find Methods
        - They create a node with a new state, but don't append it to the tree.
    """
    def find_successor_after_action(self,action):
        new_observation, reward , done , new_game_info = self.environment.step(action,observation=self.observation)
        legal_actions = self.environment.get_legal_actions(observation=new_observation)
        return Search_Node(self.environment,new_observation, parent_node=self,parent_action=action,parent_reward=reward,terminal=done,legal_actions=legal_actions)

    def find_random_successor(self):
        random_idx = random.randint(0,len(self.all_legal_actions)-1)
        action = self.all_legal_actions[random_idx]
        return self.find_successor_after_action(action)

    def find_random_unexpanded_successor(self):
        random_idx = random.randint(0,len(self.non_expanded_legal_actions)-1)
        action = self.non_expanded_legal_actions[random_idx]
        return self.find_successor_after_action(action)

    def find_rest_unexpanded_successors(self) -> list:
        generation = []
        for action in self.non_expanded_legal_actions:
            generation.append(self.find_successor_after_action(action))
        return generation

    ''' 
    Append Method
        - gets some nodes that resulted from possible actions and have not been appended to the tree and appends them
        also removes those actions from the set of unexpanded actions
    ''' 
    #! change name to add_successors
    def add_successors(self,successors: list):
        for node in successors:
            self.successors[node.get_parent_action()] = node
            self.get_non_expanded_legal_actions().remove(node.get_parent_action())

    '''
    Expand Methods
        - Combines both find methods with append. It's a way to create new nodes that result after the current and 
        appends them to the tree
    '''
    def expand_random_successor(self):
        node = self.find_random_unexpanded_successor()
        self.add_successors([node])
        return node

    def expand_rest_successors(self):
        nodes = self.find_rest_unexpanded_successors()
        self.add_successors(nodes)
        return nodes

    '''
    #! add: is_leaf()
    Getters related to the tree:
        - is_root: is the node root of the tree (if doesn't have a parent node)
        - get_depth: depth of node in the tree
        - get_successors: appended node successors of current node in tree
        - get_parent_node: parent node; none if root
        - is_completely_expanded: checks if all possible node successors have been created and appended to the tree
        - get_non_expanded_legal_actions: get actions whose resulting nodes haven't been appended to the tree
    '''
    def is_root(self):
        return self.parent_node == None

    def get_depth(self):
        return self.depth

    def get_successors(self):
        list_successors = list(self.successors.values())
        list_successors.sort(key=lambda node:node.get_parent_action()) #in the tuple , the first index corresponds to the key
        return list_successors

    def get_successor_node(self,action:int):
        return self.successors[action]

    def get_parent_node(self):
        return self.parent_node

    def is_completely_expanded(self): 
        return len(self.get_non_expanded_legal_actions()) == 0

    def get_non_expanded_legal_actions(self):
        if(self.non_expanded_legal_actions is None):
            self.non_expanded_legal_actions = self.get_all_legal_actions()
        return self.non_expanded_legal_actions

    '''
    Getters about the state:
        - get_player: returns a Player class
        - get_current_observation: observation of environment in the node's state
        - render: ...
        - is_terminal: checks if state of node is terminal
        - get_winner: if the node is terminal get winner according to Player class which also accounts for TIEs
        - get_all_legal_actions: all actions possible in the current state of node (independently of successors)
        - get_mask: get np.array or list with 1s in possible legal actions and 0s in impossible legal actions from this node's state
        - get_parent_action: actions that resulted in this node from parent
    '''
    def get_player(self):
        return self.environment.get_current_player(observation=self.observation)

    def get_current_observation(self):
        return self.environment.get_current_observation(observation=self.observation)

    def render(self):
        return self.environment.get_current_observation(observation=self.observation)

    def is_terminal(self): #! change name to is_terminal_state()
        if(self.terminal != None):
            return self.terminal
        else:
            raise ValueError("Impossible to get if is terminal state")

    def get_winner_player(self):
        if not self.is_terminal():
            raise ValueError("Node is not Terminal")
        return self.environment.get_winner(observation=self.observation)

    def get_all_legal_actions(self):
        if(self.all_legal_actions is not None):
            return self.all_legal_actions
        else:
            raise ValueError("Impossible to get all legal actions")

    def get_mask(self):
        return self.environment.get_mask(observation=self.observation)

    def get_parent_action(self):
        return self.parent_action

    def get_parent_reward(self):
        return self.parent_reward

    def get_path_reward(self):
        return self.path_reward

      