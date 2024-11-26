import torch
import torch.nn as nn

class Teacher(nn.Module):
    
    def __init__(self):
        super(Teacher, self).__init__()
        self.activation = nn.ReLU()
        self.net = nn.Sequential(
            nn.Linear(100, 75),
            self.activation,
            nn.Linear(75, 50),
            self.activation,
            nn.Linear(50, 25),
            self.activation,
            nn.Linear(25, 1)
        )
    
    def forward(self, x):
        return self.net(x)
    
    def init_weights(self, mu=0, sigma=1):
        # Initialize weights according to a normal distribution
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(mu, sigma)
                layer.bias.data.normal_(mu, sigma)
                


class StudentU(nn.Module):
    
    def __init__(self):
        super(StudentU, self).__init__()
        self.activation = nn.ReLU()
        self.net = nn.Sequential(
            nn.Linear(100, 10),
            self.activation,
            nn.Linear(10, 1)
        )
    
    def forward(self, x):
        return self.net(x)
    
    def init_weights(self):
        # Xavier initialization
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
    

class StudentE(nn.Module):
    
    def __init__(self):
        super(StudentE, self).__init__()
        self.activation = nn.ReLU()
        self.net = nn.Sequential(
            nn.Linear(100, 75),
            self.activation,
            nn.Linear(75, 50),
            self.activation,
            nn.Linear(50, 25),
            self.activation,
            nn.Linear(25, 1)
        )
    
    def forward(self, x):
        return self.net(x)
    
    def init_weights(self):
        # Xavier initialization
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)    
        
    
class StudentO(nn.Module):
        
    def __init__(self):
        super(StudentO, self).__init__()
        self.activation = nn.ReLU()
        self.net = nn.Sequential(
            nn.Linear(100, 200),
            self.activation,
            nn.Linear(200, 200),
            self.activation,
            nn.Linear(200, 200),
            self.activation,
            nn.Linear(200, 100),
            self.activation,
            nn.Linear(100, 1),
        )
        
    def forward(self, x):
        return self.net(x)
    
    def init_weights(self):
        # Xavier initialization
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)