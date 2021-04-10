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
        #* expand random node
        succ_node = self.search.current_node.expand_random_successor()
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