import torch
import torch.nn as nn
import torch.nn.functional as F

#
#       Multi Layer Perceptron
#
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1, activation=nn.ReLU, dropout=0.0):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.activation = activation()
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self.activation,
                nn.Dropout(dropout)
            ) for _ in range(n_layers)
        ])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        self.softmax = nn.Softmax(-1)

        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.init_weights()
        
    def forward(self, x):
        x = self.input_layer(x)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        x = self.output_layer(x)
        x = self.softmax(x)
        return x
    
    def init_weights(self):
        for layer in self.hidden_layers:
            nn.init.xavier_normal_(layer[0].weight)
            nn.init.zeros_(layer[0].bias)
        nn.init.xavier_normal_(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)




#
#      Convolutional Neural Network
#
class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        activation=nn.ReLU,
    ):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation()
        self.max_pool = nn.MaxPool2d(2)
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.init_weights()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.max_pool(x)
        return x

    def init_weights(self):
        for name, param in self.conv.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)


class CNN(nn.Module):

    def __init__(
        self,
        pre_hidden_dim,
        hidden_dim,
        output_dim,
        n_mlp_layers=1,
        conv_activation=nn.ReLU,
        mlp_activation=nn.ReLU,
        dropout=0.0,
    ):
        super(CNN, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            ConvBlock(1, 16, 2, 1, 1, conv_activation),
            ConvBlock(16, 32, 2, 1, 1, conv_activation),
            ConvBlock(32, 64, 2, 1, 1, conv_activation),
        )

        # Fully connected layers
        self.mlp = MLP(pre_hidden_dim, hidden_dim, output_dim, n_mlp_layers, mlp_activation, dropout)

        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return F.softmax(x, dim=1)


