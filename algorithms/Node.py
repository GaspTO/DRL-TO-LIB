import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
import random
from environments.Gomoku import GomokuEnv
from environments.K_Row import K_RowEnv, EMPTY, FIRST_PLAYER, SECOND_PLAYER, TIE

MCTS_FIRST_PLAYER = 1
MCTS_SECOND_PLAYER = -1
MCTS_TIE = 0

class MCTSNode():
    def __init__(self,parent_node=None,parent_action=None,terminal=None,legal_actions=None,content: dict = {}):
        self.parent_node = parent_node
        self.parent_action = parent_action
        self.successors = []
        self.depth = 0 if parent_node is None else self.parent_node.depth + 1
        self.terminal = terminal
        self.all_legal_actions = legal_actions
        self.non_expanded_legal_actions = legal_actions
        self.content = content

 
    def find_successor_after_action(self,action, content = {}):
        raise NotImplemented()

    def find_random_unexpanded_successor(self,content = {}):
        random_idx = random.randint(0,len(self.non_expanded_legal_actions)-1)
        action = self.non_expanded_legal_actions[random_idx]
        return self.find_successor_after_action(action,content)

    def find_rest_unexpanded_successors(self,content = {}):
        generation = []
        for action in self.non_expanded_legal_actions:
            generation.append(self.find_successor_after_action(action,content))
        return generation
        
    def append_successors_to_node(self,successors: list):
        for node in successors:
            self.successors.append(node)
            self.get_non_expanded_legal_actions().remove(node.get_parent_action())

    def expand_random_successor(self,content = {}):
        node = self.find_random_unexpanded_successor(content)
        self.append_successors_to_node([node])
        return node

    def expand_rest_successors(self, content = {}):
        nodes = self.find_rest_unexpanded_successors(content)
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

    def get_content(self):
        return self.content

    def get_content_item(self,key):
        if key in self.content:
            return self.content[key]
        return None

    def set_content_item(self,key,value):
        self.content[key] = value

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

    def who_whon(self):
        #Return -1 if tie, else return the num of the player who won
        #raise error if asked when it isn't terminal
        raise NotImplemented()


class Gomoku_MCTSNode(MCTSNode):
    def __init__(self,state, parent_node=None, parent_action=None, terminal=None,legal_actions=None,content: dict = {}):
        super().__init__(parent_node=parent_node, parent_action=parent_action, terminal=terminal, legal_actions=legal_actions, content = content)
        self.set_content_item('state',state)

        
    def find_successor_after_action(self,action,content = {}):
        new_state = self.get_content_item('state').act(action)
        terminal = new_state.board.is_terminal()
        legal_actions = new_state.board.get_legal_action()
        return Gomoku_MCTSNode(new_state,parent_node = self,parent_action=action,\
            terminal = terminal, legal_actions = legal_actions,content = content)

    def get_all_legal_actions(self):
        if(self.all_legal_actions != None):
            return self.all_legal_actions
        else:
            self.all_legal_actions = self.get_content_item('state').board.get_legal_action()
            self.non_expanded_legal_actions = self.all_legal_actions
            return self.all_legal_actions

        
    

class K_Row_MCTSNode(MCTSNode):
    def __init__(self,state, parent_node=None, parent_action=None, terminal=None,legal_actions=None,content: dict = {}):
        super().__init__(parent_node=parent_node, parent_action=parent_action, terminal=terminal, legal_actions=legal_actions, content = content)
        self.set_content_item('state',state)
        self.terminal = state.is_terminal()
        
    def find_successor_after_action(self,action,content = {}):
        new_state = self.get_content_item('state').act(action)
        terminal = new_state.is_terminal()
        legal_actions = list(new_state.get_valid())
        return (K_Row_MCTSNode(new_state,parent_node = self,parent_action=action,terminal = terminal, legal_actions = legal_actions,content = content)
                ,0)

    def get_all_legal_actions(self):
        if(self.all_legal_actions != None):
            return self.all_legal_actions
        else:
            self.all_legal_actions = list(self.get_content_item('state').get_valid())
            self.non_expanded_legal_actions = self.all_legal_actions
            return self.all_legal_actions

    def is_terminal(self):
        if self.terminal is None:
            self.terminal = self.get_content_item('state').is_terminal()
        else:
            return self.terminal

    def who_won(self):
        if not self.is_terminal():
            raise ValueError("Node is not Terminal")
        else:
            state =  self.get_content_item('state')
            if(state.winner == FIRST_PLAYER):
                return MCTS_FIRST_PLAYER
            elif(state.winner == SECOND_PLAYER):
                return MCTS_SECOND_PLAYER
            elif(state.winner == TIE):
                return MCTS_TIE
            else:
                raise ValueError("Some weird value was returned in who_won")

    


''' Teste Gomoku
env = GomokuEnv('black','random',9)
gom_node = Gomoku_Node(env.state)
actions = gom_node.get_all_legal_actions()
print(actions)
act0 = actions[0]
gom_node2 = gom_node.find_successor_after_action(act0)
print(gom_node2)
actions = gom_node2.get_all_legal_actions()
print(actions)
'''

'''
env = K_RowEnv(board_shape=3, target_length=3)
env.reset()
env.step(3)
env.step(4)
env.step(0)
env.step(2)
env.step(6)
print(env.render())
krow_node = K_Row_Node(env.state)
print(krow_node.get_all_legal_actions())
'''
