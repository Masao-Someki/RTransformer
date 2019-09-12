# pytorch implementation of residual normalization layer

import torch
import torch.nn as nn


class ResidualNorm(nn.Module):
    def __init__(self, net, dropout=0.2):
        self.net = net
        self.norm = nn.LayerNorm()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, *args):
        x = self.norm(inputs)
        x = self.net(x, *args)
        x = inputs + self.dropout(x)
        return x
