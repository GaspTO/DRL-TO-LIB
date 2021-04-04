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

class MCTS_Agent(Agent):
    '''
    Base class for agents based on MCTS.
    It runs a normal MCTS
    '''
    def __init__(self,environment,n_iterations,exploration_weight=1.0):
        super().__init__(environment)
        self.n_iterations = n_iterations
        self.exploration_weight = exploration_weight
        self.sel_fn = None
        self.exp_fn = None
        self.sim_fn = None
        self.bkp_fn = None
        self.score_fn = None

    def play(self,observation=None):
        if observation is None: observation = self.environment.get_current_observation()
        search = MCTS_Search(self.environment, observation = observation, exploration_weight = self.exploration_weight)
        search.run_n_playouts(self.n_iterations, sel_fn=self.sel_fn, exp_fn=self.exp_fn, sim_fn=self.sim_fn, bkp_fn=self.bkp_fn)
        probs = search.get_action_probabilities(score_node_fn=self.score_fn)
        return probs.argmax()

    def set_sel_fn(self,sel_fn): self.sel_fn = sel_fn
    def set_exp_fn(self,exp_fn): self.exp_fn = exp_fn
    def set_sim_fn(self,sim_fn): self.sim_fn = sim_fn
    def set_bkp_fn(self,bkp_fn): self.bkp_fn = bkp_fn
    def set_score_fn(self,score_fn): self.score_fn = score_fn


'''
    Like MCTS_Agent but
    - selection_tactic:  U + Q =  Exploration * network_probability * sqrt(parent_visits)/(1 + node_visits)   + opponent_losses/(parent_visits + 1)
    - exploration_tactic:  Expands everynode and give them a network_probability
'''
class MCTS_Simple_RL_Agent(MCTS_Agent):

    def __init__(self,environment,n_iterations,network,device,exploration_weight = 1.0):
        super().__init__(environment,n_iterations,exploration_weight=exploration_weight)
        self.network = network
        self.device = device
        def _selection_tactic(mcts):
            sqrt_N = sqrt(mcts.current_node.num_chosen_by_parent)
            def puct(node):
                assert node.num_chosen_by_parent == node.num_losses + node.num_draws + node.num_wins
                opponent_losses = node.num_losses + 0.5 * node.num_draws
                U = mcts.exploration_weight * node.p * sqrt_N /(1 + node.num_chosen_by_parent)
                Q = opponent_losses/(node.num_chosen_by_parent + 1)
                return U + Q
            max_node =  max(mcts.current_node.get_successors(), key=puct)
            return max_node

        def _expansion_tactic(mcts):
                nodes = mcts.current_node.expand_rest_successors()
                current_board = mcts.current_node.get_current_observation()
                x = torch.from_numpy(current_board).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    p = self.network(x,torch.tensor(mcts.current_node.get_mask()),False)
                    p = torch.softmax(p,dim=1)
                for node in nodes:
                    node.p = p[0][node.parent_action]
                    node.belongs_to_tree = True
                random_idx = random.randint(0,len(nodes)-1)
                return nodes[random_idx]

        self.set_sel_fn(_selection_tactic)
        self.set_exp_fn(_expansion_tactic)


'''
    Like MCTS_Simple_RL_Agent but
    - selection tactic described in 'Combining Q-Learning and Search with Amortized Value Estimates'
'''
class MCTS_Exploratory_RL_Agent(MCTS_Simple_RL_Agent):
    
    def __init__(self,environment,n_iterations,network,device,exploration_weight = 1.0):
        super().__init__(environment,n_iterations,network,device,exploration_weight=exploration_weight)
        def _selection_tactic(mcts):
            log_N_parent= log(mcts.current_node.num_chosen_by_parent)
            def SAVE_uct(node):
                assert node.num_chosen_by_parent == node.num_losses + node.num_draws + node.num_wins
                opponent_losses = node.num_losses + 0.5 * node.num_draws
                U = mcts.exploration_weight * sqrt(log_N_parent/node.num_chosen_by_parent)
                Q = (opponent_losses+node.p)/(node.num_chosen_by_parent + 1)
                return U + Q
            max_node =  max(mcts.current_node.get_successors(), key=SAVE_uct)
            return max_node

        self.set_sel_fn(_selection_tactic)

        


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
