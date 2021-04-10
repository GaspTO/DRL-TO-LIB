from abc import abstractmethod, abstractproperty
from agents.tree_agents.Tree_Search_Iteration import Tree_Search_Iteration
from agents.Agent import Agent

class Basic_Tree_Search_Iteration(Tree_Search_Iteration):
    def __init__(self,environment,search_fn,search_stop_condition_fn,intermediate_fn,backtrack_fn):
        super().__init__(environment)
        self.search_stop_condition_fn = search_stop_condition_fn
        self.intermediate_fn = intermediate_fn
        self.backtrack_fn = backtrack_fn
        

    '''
    play
        chooses action for observation
    '''
    def play(self,observation = None):
        if observation is None: self.observation = self.environment.get_current_observation() 
        self._initialize_search()
        self._run_n_playouts(self.n_iterations)
        return self._get_best_action()

    def _initialize_search():
        pass

    def _run_n_playouts(self,n):
        for _ in range(n):
            self.do_search_phase()
            self.do_collection_phase()
            self.do_backtrack_phase()

    def _get_best_action(self):
        pass
            
    '''
    do_search_phase
        runs a search algorithm until some condition is met
    '''
    def do_search_phase(self):
        while not self._search_condition():
            self._search.step()
        
    def _search_stop_condition():
        pass
    '''
    do_intermediate_phase
        after doing the search, there's usually some intermediate things we'd like to do
        this method serves to stack those methods. It might be rollouts from leaves, for e.g. 
        Or append a new node.
    '''
    @abstractmethod
    def do_intermediate_phase(self):
        pass

    '''
    do_backtrack_phase
        uses the the current search tree to update its nodes with values that will aid
        the future searches. 
    '''
    @abstractmethod
    def do_backtrack_phase(self):
        pass

    
    

