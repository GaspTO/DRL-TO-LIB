from agents.Agent import Agent
from agents.Neural_Agent import Neural_Agent

class Abstract_Network_Search_Agent(Agent):
    def __init__(self,environment,network:Neural_Agent,device):
        super().__init__(environment)
        self.network = network
        self.device = device

    def set_network(self,network:Neural_Agent):
        if not isinstance(network,Neural_Agent):
            raise ValueError("network parameter is a neural agent")
        self.network = network

