
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
import numpy

class Net(nn.Module):
	def __init__(self):
		super().__init__()
		self.seq = nn.Sequential(
      nn.Linear(28 * 28, 64),
      nn.ReLU(),
      nn.Linear(64, 3))
		

	def forward(self, x):
		embedding = self.encoder(x)
		return embedding

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
		return optimizer

	def training_step(self, train_batch, batch_idx):
		x, y = train_batch
		x = x.view(x.size(0), -1)
		z = self.encoder(x)    
		x_hat = self.decoder(z)
		loss = F.mse_loss(x_hat, x)
		self.log('train_loss', loss)
		return loss

	def validation_step(self, val_batch, batch_idx):
		x, y = val_batch
		x = x.view(x.size(0), -1)
		z = self.encoder(x)
		x_hat = self.decoder(z)
		loss = F.mse_loss(x_hat, x)
		self.log('val_loss', loss)



weight = torch.rand(1) - 0.5
weight.requires_grad = True
optimizer = torch.optim.Adam([weight], lr=0.1)
def eq(x,y):
    return max(x,y)

def AND_op(x_tensor,y_tensor):
    return x_tensor * y_tensor

def OR_op(x_tensor,y_tensor):
    return max(x_tensor,y_tensor)


    

def AND_OR_unit(x_tensor,y_tensor):
    return

def net(x,y,w=weight):
    gate = torch.sigmoid(w)
    x = torch.tensor(x)
    y = torch.tensor(y)
    AND = AND_op(x,y)
    OR = torch.max(x,y)
    gatedAND = gate * AND
    gatedOR = (1-gate)*OR
    output = gatedAND + gatedOR
    print("output=" + str(output))
    optimize(output,x,y,weight)


def optimize(output,x,y,w=weight,optim=optimizer):
    error = (torch.tensor([eq(x,y)])-output)**2 + (max(weight))
    error.backward()
    optim.step()

for i in range(10):
    for a in range(0,11):
        a = a/10
        for b in range(0,11):
            b = b/10
            net(a,b)



print("ok Teste")

    
