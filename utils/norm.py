# pytorch implementation of residual normalization layer

import torch
import torch.nn as nn


class ResidualNorm(nn.Module):
    def __init__(self, net, feature, dropout=0.2, is_rnn=False):
        super(ResidualNorm, self).__init__()
        self.net = net
        self.layernorm = nn.LayerNorm(feature)
        self.is_rnn = is_rnn
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, **kwargs):
        x = self.layernorm(inputs)
        if self.is_rnn:
            x, _ = self.net(x, **kwargs)
        else:
            x = self.net(x, **kwargs)
        x = inputs + self.dropout(x)
        return x
