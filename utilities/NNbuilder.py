import torch
import torch.tensor as tensor
import torch.nn as nn

class NNbuilder(nn.Module):




    def conv_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1

    def __init__(self,architecture,input_dim,output_size):
        super(NNbuilder, self).__init__()

        self.units = {
            "Linear": self.add_Linear_layer,
            "Conv2d": self.add_Conv2d_layer,
            "MaxPool2d": self.add_MaxPool2d,
        }

        self.activations = {
            "ReLU": nn.ReLU,
            "Sigmoid": nn.Sigmoid,
            "Tanh": nn.Tanh,
            "Softmax": nn.Softmax #create anonymous class
        }

        self.model = nn.ModuleList()
        self.secondary_operations = []
        self.architecture = architecture    
        self.input_dim = input_dim
        self.output_size = output_size
        self.last_dim = self.input_dim
        self.build_from_config()
            
        
    def build_from_config(self):
        for u in self.architecture:
            self.units[u[0]](*u[1:])
        self.add_output_layer()


    def add_to_model(self,unit,new_dim = None,secondary_operation=None):
        self.model.append(unit)
        self.secondary_operations.append(secondary_operation)
        if(new_dim is not None):
            self.last_dim = new_dim

    def add_Linear_layer(self,hidden_nodes: int, activation: str = None, act_arguments: dict = {}):
        secondary_op = None
        if(len(self.last_dim) != 2):
            secondary_op = [lambda x: x.reshape(x.size(0),-1)]
        self.add_to_model(nn.Linear(torch.prod(self.last_dim[1:]).item(),hidden_nodes),\
            torch.tensor([self.last_dim[0],hidden_nodes]),secondary_op)
        if(activation is not None):
            self.add_to_model(self.activations[activation](**act_arguments))       

    #def add_Conv1d_layer(self,out_channels,kernel_size,stride, activation_function):

    def add_Conv2d_layer(self,out_channels,kernel_size,stride,activation: str = None):
        if(len(self.last_dim)!=4):
            raise ValueError("Conv needs a 4-D input")
        
        h = NNbuilder.conv_size_out(self.last_dim[2],kernel_size,stride)
        w = NNbuilder.conv_size_out(self.last_dim[3],kernel_size,stride)
        self.add_to_model(nn.Conv2d(self.last_dim[1],out_channels,kernel_size,stride),torch.tensor([self.last_dim[0],out_channels,h,w]))
        if(activation is not None):
            self.add_to_model(self.activations[activation]())

    def add_MaxPool2d(self,kernel_size,stride):
        self.add_to_model(torch.nn.MaxPool2d(kernel_size,stride))

    def add_output_layer(self):
        self.add_Linear_layer(self.output_size,"Softmax",{"dim":1})

    def forward(self,x):
        for operation_set in zip(self.secondary_operations,self.model):
            if(operation_set[0] is not None):
                for second_op in operation_set[0]:
                    x = second_op(x)
            x = operation_set[1](x)
        return x





'''
config = {
    "input_dim": (1,2,20,20),
    "architecture": (("Conv2d",2,2,1,"ReLU"),
        ("Conv2d",5,2,1,"ReLU"),
        ("Linear",5,"ReLU"),
        ("Linear",15,"ReLU"),
        ("Linear",1,"ReLU"))
}    
inpute = torch.tensor([1,2,20,20])
#a = NNbuilder(input_dim=inpute)
a = NNbuilder(config=config)

print(a(torch.rand(tuple(inpute))))
'''





