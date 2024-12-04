import torch
import torch.nn as nn
import torch.nn.functional as F


class ResConv1d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ResConv1d, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.norm = nn.BatchNorm1d(out_channels)
        
        self.net = nn.Sequential(
            self.conv1,
            self.act1,
            self.conv2,
            self.act2
        )
        
    def forward(self, x):
        return self.net(x) + x



class Net(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Net, self).__init__()
        
        # 1d convolutional layer with residual connection for each input dimension
        self.resconv1d_arr = nn.ModuleList()
        for _ in range(input_dim):
            self.resconv1d_arr.append(ResConv1d(1, 1, 3, padding=1))
        
        # fully connected layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        self.fcnn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        x = self.fc1(x)
        
        # apply 1d convolutional layer with residual connection for each input dimension
        x = x.unsqueeze(1)
        for i in range(x.size(2)):
            x[:, :, i] = self.resconv1d_arr[i](x[:, :, i])
        x = x.squeeze(1)
        
        x = self.fcnn(x)
        
        return x
            
        

