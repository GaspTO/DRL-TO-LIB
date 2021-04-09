'''
This node is made to be as simple as possible on trees (one parent_node)
'''
class Proxy_Node():
    def __init__(self,original_node=None,proxy_parent_node=None,proxy_visited=False):
        self.original_node = original_node
        self.proxy_parent_node = proxy_parent_node
        self.proxy_visited = proxy_visited
        self.proxy_successors = []

    def add_proxy_child(self,child_node):
        self.successors.append(child_node)

    def get_original_node(self):
        return self.original_node

    def get_proxy_successors(self):
        return self.proxy_successors

    def get_proxy_parent_node(self):
        return self.proxy_parent_node

    def is_proxy_root(self):
        return self.proxy_parent_node is None

    def is_proxy_leaf(self):
        return len(self.proxy_successors) == 0

    def set_proxy_visited(self):
        self.proxy_visited = True