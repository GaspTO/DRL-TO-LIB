from agents.networks.Network_Utils import get_conv_out
from agents.networks.Neural_Network import Neural_Network
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv2d_size_out(size, kernel_size, stride,padding=0):
            return ((size + 2*padding) - (kernel_size - 1) - 1) // stride  + 1

#! device

class V_Network(Neural_Network):
	def __init__(self,device,height,width,hidden_nodes=300):
		super().__init__()
		'''
		self.convw = conv2d_size_out(conv2d_size_out(width,kernel_size=3,stride=1),kernel_size=3,stride=1)
		self.convh = conv2d_size_out(conv2d_size_out(height,kernel_size=3,stride=1),kernel_size=3,stride=1)
		self.linear_input_size = self.convw * self.convh * 64 #64 = noutchannels
		self.vnet = nn.Sequential(
            nn.Conv2d(in_channels=2,out_channels=64,kernel_size=3,stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.linear_input_size,hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes,1),    
        )
		'''
		self.kappa = nn.Sequential(
			nn.Linear(3*3*2,hidden_nodes),
            nn.ReLU(),
			nn.Linear(hidden_nodes,hidden_nodes),
			nn.ReLU(),
            nn.Linear(hidden_nodes,hidden_nodes),
			nn.ReLU(),
			nn.Linear(hidden_nodes,1)
		)

		self.double()
		
	def forward(self, x):
		return (self.kappa(x.view(x.size(0), -1)))

	def training_step(self, batch, batch_idx):
		observations, target_values = batch
		values = self(observations)
		loss = F.mse_loss(values, target_values)
		return loss

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=2e-05)