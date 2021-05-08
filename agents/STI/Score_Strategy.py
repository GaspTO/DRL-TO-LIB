class Score_Strategy:
    def summarize(self,node):
        raise NotImplementedError

class Visit_Count_Score(Score_Strategy):
    def __init__(self,temperature=1):
        super().__init__()
        self.temperature = temperature
    
    def summarize(self, node):
        if node.N == 0:
            return 0.  
        return (node.N) / node.get_parent_node().N


class Win_Ratio_Score(Score_Strategy):
    def __init__(self):
        super().__init__()
    
    def summarize(self, node):
        if node.N == 0:
            return 0.  
        return (node.W) / node.N

