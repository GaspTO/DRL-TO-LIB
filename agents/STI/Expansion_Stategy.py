from environments.core.Players import Players, Player, IN_GAME, TERMINAL, TIE_PLAYER_NUMBER

""" Expand """
class Expansion_Strategy:
    def __init__(self,search):
        self.search = search

    def expand(self):
        pass

    def reset(self):
        pass

    
class One_Successor_Rollout(Expansion_Strategy):
    def __init__(self,search):
        super().__init__(search)

    def expand(self):
        succ_node = self.expand_random_node()
        winner_player = self.simulate_from_random_node(succ_node)
        self.backpropagate_from_rollout_till_succ(succ_node, winner_player)
        return winner_player

    def expand_random_node(self):
        #* expand random node
        succ_node = self.search.current_node.expand_random_successor()
        succ_node.belongs_to_tree = True
        assert succ_node.num_chosen_by_parent == 0
        assert succ_node.num_draws + succ_node.num_losses + succ_node.num_wins == 0
        succ_node.num_chosen_by_parent = 1
        return succ_node

    def simulate_from_random_node(self,succ_node):
        #* estimation: random rolllout
        rollout_node = succ_node
        while not rollout_node.is_terminal():
            rollout_node = rollout_node.find_random_unexpanded_successor()
            rollout_node.belongs_to_tree = False
        winner_player = rollout_node.get_winner()
        return winner_player

    def backpropagate_from_rollout_till_succ(self,succ_node,winner_player):
        #* backprop from rollout
        if TIE_PLAYER_NUMBER == winner_player.get_number():
            succ_node.num_draws += 1
        elif succ_node.get_current_player().get_number() == winner_player.get_number():
            succ_node.num_wins += 1
        else:
            succ_node.num_losses += 1


