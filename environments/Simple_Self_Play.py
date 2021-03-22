from environments.Simple_Playground_Env import Simple_Playground_Env
from agents.Base_Agent import Base_Agent
import copy

class Simple_Self_Play(Simple_Playground_Env):
	def __init__(self,episodes_to_update,environment=None):
		self.ep_2_upd = episodes_to_update
		self.current_ep = 0
		self.original_agent = None
		super().__init__(environment)

	def add_agent(self, agent):
		if not isinstance(agent,Base_Agent):
			raise ValueError('agent needs to be a Base_Agent')
		self.original_agent = agent
		self.update_adversary_agent()
			
	def update_adversary_agent(self):
		if self.original_agent is None:
			raise ValueError('agent needs to be added')
		old_agent =  self.get_adversary_agent()
		new_agent = self.original_agent.clone()
		super().set_adversary_agent(new_agent)

	def step(self,action,zero_sum = True, all_info = False):
		next_state, total_reward, done, total_info = super().step(action,zero_sum=zero_sum,all_info=all_info)
		if done == True: self.current_ep += 1
		if self.current_ep % self.ep_2_upd == 0 and done == True:
			self.update_adversary_agent()
		return next_state, total_reward, done, total_info

    

