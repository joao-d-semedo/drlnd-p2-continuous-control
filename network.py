import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    
    def __init__(self, sizes, activation=nn.ReLU, output_activation=None, seed=None):
        super().__init__()

        if seed is not None:
            self.seed = torch.manual_seed(seed)
        
        self.layers = []
        for i in range(len(sizes)-2):
            self.layers += [nn.Linear(sizes[i], sizes[i+1]), activation()]
        self.layers.append( nn.Linear(sizes[-2], sizes[-1]) )

        if output_activation is not None:
            self.layers.append(output_activation())
        
        self.layers = nn.ModuleList(self.layers)
        
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        for layer in self.layers:
            x = layer(x)
        return x

