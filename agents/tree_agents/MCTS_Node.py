import random
import copy
from environments.core.Players import Players, Player, IN_GAME, TERMINAL


class MCTS_Node():
    def __init__(self,environment_interface,observation,parent_node=None,parent_action=None,terminal=None,legal_actions=None):
        self.environment = environment_interface
        self.observation = observation
        self.parent_node = parent_node
        self.parent_action = parent_action
        self.successors = []
        self.depth = 0 if parent_node is None else self.parent_node.depth + 1
        self.terminal = terminal if terminal != None else self.environment.is_terminal(observation=observation)
        self.all_legal_actions = list(legal_actions) if legal_actions is not None else list(self.environment.get_legal_actions(observation=observation))
        self.non_expanded_legal_actions = copy.deepcopy(self.all_legal_actions)
        
        '''
        Auxiliary
            #todo there's no reason to have these here, MCTS_Node should be a normal node
        '''
        self.num_wins = 0
        self.num_losses = 0
        self.num_draws = 0
        self.num_chosen_by_parent = 0
        self.belongs_to_tree = False
        
    """
    Find Methods
        - They create a node with a new state, but don't append it to the tree.
    """
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

    ''' 
    Append Method
        - gets some nodes that resulted from possible actions and have not been appended to the tree and appends them
        also removes those actions from the set of unexpanded actions
    ''' 
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
    Getter Methods:
        - is_terminal: checks if state of node is terminal
        - is_completely_expanded: checks if all possible node successors have been created and appended to the tree
        - get_non_expanded_legal_actions: get actions whose resulting nodes haven't been appended to the tree
        - is_root: is the node root of the tree (if doesn't have a parent node)
        - get_depth: depth of node in the tree
        - get_successors: appended node successors of current node in tree
        - get_all_legal_actions: all actions possible in the current state of node (independently of successors)
        - get_parent_node: parent node; none if root
        - get_parent_action: actions that resulted in this node from parent
        - get_winner: if the node is terminal get winner according to Player class which also accounts for TIEs
        - get_current_player: returns a Player class
        - get_current_observation: observation of environment in the node's state
        - get_mask: get np.array or list with 1s in possible legal actions and 0s in impossible legal actions from this node's state
        - render: ...
    '''
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
        if(self.all_legal_actions is not None):
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

    def get_mask(self):
        return self.environment.get_mask(observation=self.observation)

    def render(self):
        return self.environment.get_current_observation(observation=self.observation)

      