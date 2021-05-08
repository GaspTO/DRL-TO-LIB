""" Backup"""
class Backup_Strategy:
    def update(self,node_to_update):
        raise NotImplementedError


class Backup_W_N_one_successor(Backup_Strategy):
    '''
    Does three things:
        adds subtree_total_R to W
        and propagates subtree_total_R to parent_node
        resets subtree_total_R of node back to 0.
    '''
    def update(self,node):
        parent_node = node.get_parent_node()
        node.N += 1
        if parent_node is not None:
            if node.get_parent_node().get_player() == node.get_player():
                delta_W = node.subtree_total_R +  node.get_parent_reward()
                node.W += delta_W
                parent_node.subtree_total_R += delta_W
            else:
                delta_W = node.subtree_total_R +  -1*node.get_parent_reward()
                node.W += delta_W
                parent_node.subtree_total_R += -1*delta_W
        else:
            node.W += node.subtree_total_R
        
        node.subtree_total_R = 0

        



