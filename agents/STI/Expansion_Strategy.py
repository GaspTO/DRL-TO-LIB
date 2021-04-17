from environments.core.Players import Players, Player, IN_GAME, TERMINAL, TIE_PLAYER_NUMBER
import torch
import random

""" Expand """
class Expansion_Strategy:
    def __init__(self):
        pass

    def update_tree(self,node):
        pass

    def reset(self):
        pass

    
class One_Successor_Rollout(Expansion_Strategy):
    def __init__(self):
        super().__init__()

    def update_tree(self,node):
        if not node.is_terminal():
            succ_node = self._expand_new(node)
            winner_player = self._simulate_from_random_node(succ_node)
            self._backpropagate(succ_node, winner_player)
        else:
            winner_player = node.get_winner()
            self._backpropagate(node,winner_player)

    def _expand_new(self,node):
        #* expands random node and rollout
        succ_node = node.expand_random_successor()
        succ_node.belongs_to_tree = True
        return succ_node

    def _simulate_from_random_node(self,succ_node):
        #* estimation: random rolllout
        rollout_node = succ_node
        while not rollout_node.is_terminal():
            rollout_node = rollout_node.find_random_unexpanded_successor()
            rollout_node.belongs_to_tree = False
        winner_player = rollout_node.get_winner()
        return winner_player

    def _backpropagate(self,node,winner):
        while not node.is_root():
            self._update_statistic(node,winner)
            node = node.get_parent_node()
        assert node.is_root()
        self._update_statistic(node,winner)

    def _update_statistic(self,node,winner):
            node.num_chosen_by_parent += 1
            if TIE_PLAYER_NUMBER == winner.get_number():
                node.num_draws += 1
            elif node.get_current_player().get_number() == winner.get_number():
                node.num_wins += 1
            else:
                node.num_losses += 1



class Network_One_Successor_Rollout(One_Successor_Rollout):
    def __init__(self,network,device):
        super().__init__()
        self.network = network
        self.device = device

    def update_tree(self,node):
        if not node.is_terminal():
            succ_node = self._expand_new(node)
            winner_player = self._simulate_from_random_node(succ_node)
            self._backpropagate(succ_node, winner_player)
        else:
            winner_player = node.get_winner()
            self._backpropagate(node,winner_player)

    def _expand_new(self,node):
        ''' expand all and adds node.p '''
        nodes = node.expand_rest_successors()
        current_board = node.get_current_observation()
        x = torch.from_numpy(current_board).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            p = self.network(x,torch.tensor(node.get_mask()),False)
            p = torch.softmax(p,dim=1)
        for node_child in nodes:
            #* real values
            if node_child.is_terminal():
                if node_child.get_winner() == node_child.get_current_player():
                   node_child.p = 1
                elif node_child.get_winner() == node_child.get_current_player():
                    node_child.p = -1
                else:
                   node_child.p = 0
            else:
                node_child.p = p[0][node_child.parent_action].item()
                node_child.belongs_to_tree = True
        random_idx = random.randint(0,len(nodes)-1)
        return nodes[random_idx]


