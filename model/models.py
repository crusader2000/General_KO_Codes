import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class g_Full(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super(g_Full, self).__init__()
        
        self.input_size  = input_size
        
        self.half_input_size = int(input_size/2)
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.fc1 = nn.Linear(self.input_size, self.hidden_size, bias=True).to(device)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True).to(device)
        self.fc3 = nn.Linear(self.hidden_size, self.hidden_size, bias=True).to(device)
        self.fc4 = nn.Linear(self.hidden_size, self.output_size, bias=True).to(device)
        

    def forward(self, y):
        y = y.float()
        x = F.selu(self.fc1(y))  

        x = F.selu(self.fc2(x))

        x = F.selu(self.fc3(x))
        x = self.fc4(x) +  y[:, :, :self.half_input_size]*y[:,:, self.half_input_size:]
        return x
    
class f_Full(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super(f_Full, self).__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.fc1 = nn.Linear(self.input_size, self.hidden_size, bias=True).to(device)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True).to(device)
        self.fc3 = nn.Linear(self.hidden_size, self.hidden_size, bias=True).to(device)
        self.fc4 = nn.Linear(self.hidden_size, self.output_size, bias=True).to(device)

    def forward(self, y):
        y = y.float()
        x = F.selu(self.fc1(y))
        x = F.selu(self.fc2(x))

        x = F.selu(self.fc3(x))
        x = self.fc4(x)
        return x
    
