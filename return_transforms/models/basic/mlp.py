"""
A set of methods for building MLPs with different features.
- batchnorm/layernorm
- dropout
- different activation functions
- input size, output size, hidden size
- different number of layers
"""

import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, activation, batchnorm=False, layernorm=False, dropout=0.0):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if activation == 'relu':
            self.activation = nn.ReLU
        else:
            raise NotImplementedError

        self.batchnorm = batchnorm
        self.layernorm = layernorm
        self.dropout = dropout

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.input_size, self.hidden_size))
        if self.batchnorm:
            self.layers.append(nn.BatchNorm1d(self.hidden_size))
        if self.layernorm:
            self.layers.append(nn.LayerNorm(self.hidden_size))
        self.layers.append(self.activation())
        if self.dropout > 0.0:
            self.layers.append(nn.Dropout(self.dropout))
        for i in range(self.num_layers - 1):
            self.layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            if self.batchnorm:
                self.layers.append(nn.BatchNorm1d(self.hidden_size))
            if self.layernorm:
                self.layers.append(nn.LayerNorm(self.hidden_size))
            self.layers.append(self.activation())
            if self.dropout > 0.0:
                self.layers.append(nn.Dropout(self.dropout))
        self.layers.append(nn.Linear(self.hidden_size, self.output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
