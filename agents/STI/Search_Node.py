import random
import copy
from environments.core.Players import Players, Player, IN_GAME, TERMINAL


'''
This node is made specifically to search in environments using trees (one parent node)
'''
#! change name to Environment_Node
class Search_Node():
    '''
    - initializer_fn: it's a function that's suppose to initialize a bunch of parameters that are relevant
    '''
    def __init__(self,environment_interface,observation,parent_reward=0,initializer_fn=None,parent_node=None,parent_action=None,terminal=None,legal_actions=None):
        self.environment = environment_interface
        self.observation = observation
        self.parent_node = parent_node
        self.parent_action = parent_action
        self.successors = []
        self.depth = 0 if parent_node is None else self.parent_node.depth + 1
        self.terminal = terminal if terminal != None else self.environment.is_terminal(observation=observation)
        self.all_legal_actions = list(legal_actions) if legal_actions is not None else list(self.environment.get_legal_actions(observation=observation))
        self.non_expanded_legal_actions = copy.deepcopy(self.all_legal_actions)
        ''' to initialize relevant parameters '''
        self.parent_reward = parent_reward
        self.W = 0 #summation of rewards
        self.N = 0 #number of visits
        self.delta = 0 #value to be propagated
        self.initializer_fn = initializer_fn 
        #if initializer_fn is not None: initializer_fn(self)
        
    """ 
    Find Methods
        - They create a node with a new state, but don't append it to the tree.
    """
    def find_successor_after_action(self,action):
        new_observation, reward , done , new_game_info = self.environment.step(action,observation=self.observation)
        legal_actions = self.environment.get_legal_actions(observation=new_observation)
        return Search_Node(self.environment,new_observation,parent_reward=reward,initializer_fn=self.initializer_fn,\
            parent_node = self,parent_action=action,terminal = done,legal_actions=legal_actions)

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
    def append_successors_to_node(self,successors: list):
        for node in successors:
            self.successors.append(node)
            self.get_non_expanded_legal_actions().remove(node.get_parent_action())

    '''
    Expand Methods
        - Combines both find methods with append. It's a way to create new nodes that result after the current and 
        appends them to the tree
    '''
    def expand_random_successor(self):
        node = self.find_random_unexpanded_successor()
        self.append_successors_to_node([node])
        return node

    def expand_rest_successors(self):
        nodes = self.find_rest_unexpanded_successors()
        self.append_successors_to_node(nodes)
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
        self.successors.sort(key=lambda n:n.get_parent_action())
        return self.successors

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
        - get_current_player: returns a Player class
        - get_current_observation: observation of environment in the node's state
        - render: ...
        - is_terminal: checks if state of node is terminal
        - get_winner: if the node is terminal get winner according to Player class which also accounts for TIEs
        - get_all_legal_actions: all actions possible in the current state of node (independently of successors)
        - get_mask: get np.array or list with 1s in possible legal actions and 0s in impossible legal actions from this node's state
        - get_parent_action: actions that resulted in this node from parent
    '''
    def get_current_player(self):
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

    def get_winner(self):
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

      