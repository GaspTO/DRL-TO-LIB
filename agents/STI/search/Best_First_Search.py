from agents.STI.search.Search_Interface import Search_Interface
import heapq




class Best_First_Search():
    def __init__(self,state_space,eval_fn):
        super().__init__(state_space)
        self.eval_fn = eval_fn
        
    def step(self):
        pass

    def get_frontier(self):
        return self.frontier_list

    def get_closed(self):
        return self.closed_list

    def is_frontier_empty(self):
        return len(self.frontier_list) == 0  

    def get_frontier_size(self):
        return len(self.frontier_list)

    def add_to_frontier(self,node):
        heapq()

    def get_next_node_in_frontier(self):
        
    
        

    

