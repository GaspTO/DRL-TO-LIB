import heapq

import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
sys.path.append("/home/nizzel/Desktop/Tiago/Computer_Science/Tese/DRL-TO-LIB")
from environments.Custom_K_Row import Custom_K_Row
import torch
import random
from math import sqrt, log
from agents.Agent import Agent
from agents.tree_agents.MCTS_Search import MCTS_Search

from agents.tree_agents.Search_Node import Search_Node



'''
    Like MCTS_Agent but
    - selection_tactic:  U + Q =  Exploration * network_probability * sqrt(parent_visits)/(1 + node_visits)   + opponent_losses/(parent_visits + 1)
    - exploration_tactic:  Expands everynode and give them a network_probability
'''
class MCTS_Simple_RL_Agent(MCTS_Search):
    def __init__(self,environment,n_iterations,network,device,exploration_weight = 1.0,debug=False):
        super().__init__(environment,n_iterations,exploration_weight=exploration_weight,debug=debug)
        self.network = network
        self.device = device

    '''#!REMOVE THIS 
    def _score_tactic(self,node):
        
        if node.num_chosen_by_parent == 0:
            return 0.  # avoid unseen moves
        return  0.9 * ((node.num_losses + 0.5*node.num_draws) / node.num_chosen_by_parent) + 0.1 * node.p '''
    
    '''! REMOVE THIS
    def play(self,observation = None):
        if observation is None: observation = self.environment.get_current_observation() 
        self.root =  Search_Node(self.environment,observation,initializer_fn=self.node_initializer)
        self.root.belongs_to_tree = True
        self.current_node = self.root
        self.run_n_playouts(self.n_iterations)

        #!,,,
        current_board = self.root.get_current_observation()
        x = torch.from_numpy(current_board).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            p = self.network.load_state(x).get_policy_values(True,torch.tensor(self.root.get_mask()))
        for node in self.root.get_successors():
            #!debug
            node.p = p[0][node.parent_action]
        #!....

        probs = self.get_action_probabilities()

        #! SWAP REMOVE
        def puctt(node):
            sqrt_N = sqrt(self.root.num_chosen_by_parent)
            assert node.num_chosen_by_parent == node.num_losses + node.num_draws + node.num_wins
            opponent_losses = node.num_losses + 0.5 * node.num_draws
            U = self.exploration_weight * node.p * sqrt_N /(1 + node.num_chosen_by_parent)
            Q = opponent_losses/(node.num_chosen_by_parent + 1)
            return U + Q
        current = self.current_node
        sqrt_N = sqrt(current.num_chosen_by_parent)
        self.stri = "PUCT values \n:"
        l = current.get_successors()
        l.sort(key=lambda x: x.get_parent_action())
        for node in l:
            #print("parent:"+str(node.get_parent_action()) + "    p:"+ str(node.p)+ "   (loss,draws,games):" + "(" + str(node.num_losses) + "," + str(node.num_draws) + "," + str(node.num_chosen_by_parent) + ")" +"   puct:"+str(puctt(node)) + "    U:"+ str( node.p * sqrt_N /(1 + node.num_chosen_by_parent))     +"     Q:" + str(node.num_losses + 0.5 * node.num_draws/(node.num_chosen_by_parent + 1)))
            self.stri += "\t\t\tparent:"+str(node.get_parent_action()) + "    p:"+ str(node.p)+ "   (loss,draws,games):" + "(" + str(node.num_losses) + "," + str(node.num_draws) + "," + str(node.num_chosen_by_parent) + ")" +"   puct:"+str(puctt(node)) + "    U:"+ str( self.exploration_weight * node.p * sqrt_N /(1 + node.num_chosen_by_parent))     +"     Q:" + str((node.num_losses + 0.5 * node.num_draws)/(node.num_chosen_by_parent + 1)) + "\n"
    
        #! END DEBUG 
        


        return probs.argmax()

    #! DEBUG **
    
    def _selection_tactic(self):
        log_N_vertex = log(self.current_node.num_chosen_by_parent)
        def uct(node):
                assert node.num_chosen_by_parent == node.num_losses + node.num_draws + node.num_wins
                opponent_losses = node.num_losses + 0.5 * node.num_draws
                return opponent_losses / (node.num_chosen_by_parent+1) + \
                    self.exploration_weight * sqrt(log_N_vertex / (node.num_chosen_by_parent+1))
        max_node =  max(self.current_node.get_successors(), key=uct)
        return max_node'''

    def _selection_tactic(self):
        sqrt_N = sqrt(self.current_node.num_chosen_by_parent)
        def puct(node):
            assert node.num_chosen_by_parent == node.num_losses + node.num_draws + node.num_wins
            opponent_losses = node.num_losses + 0.5 * node.num_draws
            U = self.exploration_weight * node.p * sqrt_N /(1 + node.num_chosen_by_parent)
            Q = opponent_losses/(node.num_chosen_by_parent + 1)
            return U + Q
        max_node =  max(self.current_node.get_successors(), key=puct)
        return max_node
    


    
    def _expansion_tactic(self):
            nodes = self.current_node.expand_rest_successors()
            current_board = self.current_node.get_current_observation()
            x = torch.from_numpy(current_board).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                p = self.network.load_state(x).get_policy_values(False,torch.tensor(self.current_node.get_mask()))
                p = torch.softmax(p,dim=1)
            for node in nodes:
                node.p = p[0][node.parent_action]
                node.belongs_to_tree = True
            random_idx = random.randint(0,len(nodes)-1)
            return nodes[random_idx]
    



'''
    Like MCTS_Simple_RL_Agent but
    - selection tactic described in 'Combining Q-Learning and Search with Amortized Value Estimates'
'''
class MCTS_Exploitation_RL_Agent(MCTS_Simple_RL_Agent):
    
    def __init__(self,environment,n_iterations,network,device,exploration_weight = 1.0,debug=False):
        super().__init__(environment,n_iterations,network,device,exploration_weight=exploration_weight,debug=debug)

    def _selection_tactic(self):
        log_N_parent= log(self.current_node.num_chosen_by_parent)
        def SAVE_uct(node):
            assert node.num_chosen_by_parent == node.num_losses + node.num_draws + node.num_wins
            opponent_losses = node.num_losses + 0.5 * node.num_draws
            U = self.exploration_weight * sqrt(log_N_parent/(node.num_chosen_by_parent+1))
            Q = (opponent_losses+node.p)/(node.num_chosen_by_parent + 1)
            return U + Q
        max_node =  max(self.current_node.get_successors(), key=SAVE_uct)
        return max_node





'''
    Like MCTS_Simple_RL_Agent
    + This algorithm expands the best successors opened not in a greedy way, but in an A* way. 
    In other words, it chooses the best not locally, but globally. It has the problem of not differentiating between
    Ally and Adversary. 
    To be fair this is kinda of a useless algorithm. It will expand nodes, backtrack but the backtrackting updates won't do anything
    #! REMOVE THIS ALGORITHM
'''
class MCTS_Astar_Agent(MCTS_Simple_RL_Agent):
    def __init__(self,environment,n_iterations,network,device,exploration_weight = 1.0,debug=False):
        super().__init__(environment,n_iterations,network,device,exploration_weight=exploration_weight,debug=debug)
        self.agent_heap = []

    def _selection_tactic(self):
        sqrt_N = sqrt(self.current_node.num_chosen_by_parent)
        def puct(node):
            assert node.num_chosen_by_parent == node.num_losses + node.num_draws + node.num_wins
            opponent_losses = node.num_losses + 0.5 * node.num_draws
            U = self.exploration_weight * node.p * sqrt_N /(1 + node.num_chosen_by_parent)
            Q = opponent_losses/(node.num_chosen_by_parent + 1)
            return U + Q
        max_node =  max(self.agent_heap, key=puct)
        return max_node

    def _expansion_tactic(self):
            nodes = self.current_node.expand_rest_successors()
            self.agent_heap.extend(nodes)
            if(self.current_node != self.root): self.agent_heap.remove(self.current_node)
            current_board = self.current_node.get_current_observation()
            x = torch.from_numpy(current_board).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                p = self.network(x,torch.tensor(self.current_node.get_mask()),False)
                p = torch.softmax(p,dim=1)
            for node in nodes:
                node.p = p[0][node.parent_action]
                node.belongs_to_tree = True
            random_idx = random.randint(0,len(nodes)-1)
            return nodes[random_idx]



    

'''
    Like MCTS_Simple_RL_Agent
    + This algorithm expands the best successors that is of ally. Every time it expands an ally it greedily chooses the best for the adversary.
    + This one will do multiple A* iterations, meaning that it is possible for the frontier nodes to have already been expanded in a previous iteration
    + So we proceed as normal mcts but with frontier, 1) select a node from frontier 2) if node selected hasn't been expanded, expand and choose greedily for adversary
    3) maybe simulate that node, 4) backup
'''
class MCTS_IDAstar_Agent(MCTS_Simple_RL_Agent):
    def __init__(self,environment,n_iterations,network,device,exploration_weight = 1.0,debug=False):
        super().__init__(environment,n_iterations,network,device,exploration_weight=exploration_weight,debug=debug)
        self.agent_heap = []

    def _selection_phase(self):
        while self.current_node.is_completely_expanded() and not self.current_node.is_terminal():
            self.current_node = self.selection_criteria()
            if self.debug: print("Sel:\n" + str(self.current_node.render()))

    def _selection_tactic(self):
        sqrt_N = sqrt(self.current_node.num_chosen_by_parent)
        def puct(node):
            assert node.num_chosen_by_parent == node.num_losses + node.num_draws + node.num_wins
            opponent_losses = node.num_losses + 0.5 * node.num_draws
            U = self.exploration_weight * node.p * sqrt_N /(1 + node.num_chosen_by_parent)
            Q = opponent_losses/(node.num_chosen_by_parent + 1)
            return U + Q
        for n in self.current_node.get_successors():
            heapq.heappush(self.agent_heap,(-1*puct(n),n))
        max_node =  heapq.heappop()
        return max_node

    def _expansion_tactic(self):
            ''' adversary ''' 
            adversary_nodes = self.current_node.expand_rest_successors()
            current_board = self.current_node.get_current_observation()
            x = torch.from_numpy(current_board).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                p = self.network(x,torch.tensor(self.current_node.get_mask()),False)
                p = torch.softmax(p,dim=1)
                best_node_act = p.argmax()
            for node in adversary_nodes:
                node.p = p[0][node.parent_action]
                node.belongs_to_tree = True
                if node.parent_action == best_node_act:
                    greedy_adversary_node = node

            ''' natural expansion of ally '''
            ally_nodes = greedy_adversary_node.expand_rest_successors()
            current_board = greedy_adversary_node.get_current_observation()
            x = torch.from_numpy(current_board).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                p = self.network(x,torch.tensor(greedy_adversary_node.get_mask()),False)
                p = torch.softmax(p,dim=1)
            for node in adversary_nodes:
                node.p = p[0][node.parent_action]
                node.belongs_to_tree = True    
            random_idx = random.randint(0,len(ally_nodes)-1)
            self.agent_heap = []
            return ally_nodes[random_idx]




'''
#? test:play against agent
env = Custom_K_Row(board_shape=3, target_length=3)
env.reset()
done = False
i = 0
while done == False:
    if i == 1:
        agent = MCTS_Agent(env,10000,exploration_weight=1.0)
        act = agent.play()
        print(act)
        _,_,done,_ = env.step(act)
    else:
        act = int(input())
        _,_,done,_ = env.step(act)
    i = (i + 1) % 2
'''
