import sys, os
sys.path.append("/home/nizzel/Desktop/Tiago/Computer_Science/Tese/DRL-TO-LIB")


from agents.Agent import Agent
from agents.search_agents.k_best_first_minimax.K_Best_First_Minimax_Node import K_Best_First_Minimax_Node
import numpy as np
from collections import namedtuple
import random


class K_Best_First_Minimax(Agent):
    def __init__(self,environment,expansion_st,k=1,num_iterations=None,debug=False):
        Agent.__init__(self,environment)
        self.expansion_st = expansion_st
        self.num_iterations = num_iterations
        self.k = k
        self.root = None
        self.debug = debug

    def play(self,observation=None):
        if(observation is None): observation = self.environment.get_current_observation()
        self.root = K_Best_First_Minimax_Node(self.environment,observation)
        self.search(self.root,self.num_iterations)
        return self._get_action_probabilities(self.root), {"root_node":self.root}

    def search(self,root,num_iterations):
        if root.is_terminal():
            raise ValueError("shouldn't be terminal")
        for iter_num in range(num_iterations):
            if self.debug:
                print("..................... ITER " + str(iter_num) + "   .....................")
            k_nodes = self.find_k_nodes_to_expand(root,iter_num,self.k)
            if len(k_nodes) == 0:
                return                
            else:
                self.expansion_st.expand(k_nodes)
                for node in k_nodes:
                    self.propagate_minimax_backwards(node)
            #* debug
            if self.debug:
                print("======")
                self.validate(root)
                for n in k_nodes:
                    self.unroll_actions(n)
                    print()
                print("======")
                self.debug(root)
            
    def find_k_nodes_to_expand(self,node,iteration_number,k):
        def i_successor_estimation_from_parent(succ):
            if succ.i < iteration_number:
                succ.i_non_terminal_value = succ.non_terminal_value
                succ.i = iteration_number

            if succ.get_player() == succ.get_parent_node().get_player():
                succ_i_non_terminal_value = float("-inf") if succ.i_non_terminal_value is None else succ.get_parent_reward() + succ.i_non_terminal_value
            else:
                succ_i_non_terminal_value = float("-inf") if succ.i_non_terminal_value is None else succ.get_parent_reward() + -1*succ.i_non_terminal_value
            return succ_i_non_terminal_value

        def find_non_expanded_leaf(node):
            '''
            returns non expanded leaf, removes it from iteration tree, updates ancestors values.
            or, returns None if there isn't any non-expanded leaf in current subtree
            '''
            if node.i_non_terminal_value is None:
                return None
            if not node.is_completely_expanded():
                node.i_non_terminal_value = None
                return node
            else:
                best_successor = self.rnd_max(node.get_successors(),key=i_successor_estimation_from_parent)
                if best_successor.i_non_terminal_value is None:
                    return None
                else:
                    non_expanded_leaf = find_non_expanded_leaf(best_successor)
                    if best_successor.i_non_terminal_value is None: #needs a new successor 
                        best_successor = self.rnd_max(node.get_successors(),key=i_successor_estimation_from_parent)

                    node.i_non_terminal_value = None if best_successor.i_non_terminal_value is None else i_successor_estimation_from_parent(best_successor) 
                    assert node.i_non_terminal_value != float("-inf") and node.i_non_terminal_value != float("+inf") 
                    return non_expanded_leaf
            
        k_nodes = []
        self.root.i_non_terminal_value = self.root.non_terminal_value
        for l in range(k):
            unexpanded_leaf = find_non_expanded_leaf(self.root)
            if unexpanded_leaf is None:
                return k_nodes
            else:
                k_nodes.append(unexpanded_leaf)
        return k_nodes
            

    def propagate_minimax_backwards(self,node):
        #todo you can make this more efficient by stopping when the value is not propagated
        if node is None:
            return
        elif node.is_terminal():
            raise ValueError("this function shouldn't be called on terminal nodes")
        else:
            #*minimax tree
            best_succ_node = self.rnd_max(node.get_successors(),key=lambda n: self.successor_estimation_from_parent(n)[0])
            node.value = self.successor_estimation_from_parent(best_succ_node)[0]
            #*non_terminal minimax tree
            best_non_terminal_succ_node = self.rnd_max(node.get_successors(),key=lambda n: self.successor_estimation_from_parent(n)[1])
            if best_non_terminal_succ_node.non_terminal_value is None:
                node.non_terminal_value = None
            else:
                node.non_terminal_value = self.successor_estimation_from_parent(best_non_terminal_succ_node)[1]
            #* recursion
            self.propagate_minimax_backwards(node.get_parent_node())
            
    def successor_estimation_from_parent(self,succ)->tuple:
        ''' returns tuple => (minimax value, non_terminal minimax value) '''
        if succ.get_player() == succ.get_parent_node().get_player():
            succ_value = succ.get_parent_reward() + succ.value
            succ_non_terminal_value = float("-inf") if succ.non_terminal_value is None else succ.non_terminal_value
        else:
            succ_value = succ.get_parent_reward() + -1*succ.value
            succ_non_terminal_value = float("-inf") if succ.non_terminal_value is None else -1*succ.non_terminal_value
        return (succ_value,succ_non_terminal_value)
            
    def _get_action_probabilities(self,node):
        #the length of successors is not always the action_size 'cause invalid actions don't become successors
        action_probs = np.zeros(self.environment.get_action_size()) 
        mask = self.environment.get_mask()
        num_max_values = 0
        for n in node.get_successors():
            assert action_probs[n.get_parent_action()] == 0.
            succ_estimation = self.successor_estimation_from_parent(n)[0]
            if succ_estimation == self.root.value:
                num_max_values += 1
                action_probs[n.get_parent_action()] = 1
        action_probs = action_probs * (1/num_max_values)

        return action_probs


    def rnd_max(self,list,key):
        ''' the same as max function but chooses random if there's
        several maximum '''
        max_value = float("-inf")
        max_items = []
        for item in list:
            if  max_value < key(item):
                max_value = key(item)
                max_items = [item]
            elif max_value == key(item):
                max_items.append(item)
        item =  random.choice(max_items)
        assert max_value == key(max(list,key=key))
        return item

    ''' debug '''
    def unroll_actions(self,node):
        if node is not None:
            self.unroll_actions(node.get_parent_node())
            print(node.get_parent_action(),end=",")

    def debug(self,node,recursion_level=-1,depth=0):
        tabs = "\t" * depth
        print(tabs + "a:" +str(node.parent_action) + "\tvalue:" + str(node.value) + "\tnt:" + str(node.non_terminal_value) +"  \tcompl_exp?: " + str(node.is_completely_expanded()) + "\tst_depth: " + str(node.subtree_depth))
        if recursion_level !=  0 and len(node.get_successors()) != 0:
            for n in node.get_successors():
                self.debug(n,recursion_level-1,depth+1)
            print()
           

    def validate(self,node):
        if node.is_terminal():
            node.subtree_depth = 0
            assert node.value == 0
        elif not node.is_completely_expanded():
            node.subtree_depth = 0
            return
        else:
            max_subtree_depth = 0
            max_value = float("-inf")
            max_non_terminal_value = float("-inf")
            for n in node.get_successors():
                self.validate(n)
                if max_subtree_depth < (1 + n.subtree_depth):
                    max_subtree_depth = (1 + n.subtree_depth)
                if max_value < self.successor_estimation_from_parent(n)[0]:
                    max_value = self.successor_estimation_from_parent(n)[0]
                if max_non_terminal_value < self.successor_estimation_from_parent(n)[1]:
                    max_non_terminal_value = self.successor_estimation_from_parent(n)[1]
            node.subtree_depth = max_subtree_depth
            if max_value != node.value:
                raise ValueError("Error")
            if max_non_terminal_value != node.non_terminal_value:
                if not (max_non_terminal_value == float("-inf") and node.non_terminal_value is None):
                    raise ValueError("Error2")


        

                






'''
from environments.Custom_K_Row import Custom_K_Row
from agents.search_agents.k_best_first_minimax.K_Best_First_Minimax_Expansion_Strategy import *

environment = Custom_K_Row(board_shape=3, target_length=3)
environment.step(0)
#environment.step(4)
#environment.step(2)
#environment.step(1)
#environment.step(7)
#environment.step(3)
#environment.step(5)
observation = environment.get_current_observation()
print(observation)
expansion_st = K_Best_First_All_Successors_Rollout(num_rollouts=1)
tree_policy = K_Best_First_Minimax(environment,expansion_st,k=1,num_iterations=10)

        #!sampling
action_probs, info = tree_policy.play(observation)
root_node = info["root_node"]
action = action_probs.argmax()
print(action_probs)

'''
