class Score_Strategy:
    def summarize(self,node):
        raise NotImplementedError

class Visit_Count_Score(Score_Strategy):
    def __init__(self,temperature=1):
        super().__init__()
        self.temperature = temperature
    
    def summarize(self, node):
        if node.num_visits == 0:
            return 0.  
        return (node.num_visits) / node.get_parent_node().num_visits


class Win_Ratio_Score(Score_Strategy):
    def __init__(self):
        super().__init__()
    
    def summarize(self, node):
        if node.num_visits == 0:
            return 0.  
        return (-node.total_value) / node.num_visits

