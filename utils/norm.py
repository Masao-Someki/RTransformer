# pytorch implementation of residual normalization layer

import torch
import torch.nn as nn


class ResidualNorm(nn.Module):
    def __init__(self, net, dropout=0.2, is_rnn=False):
        super(ResidualNorm, self).__init__()
        self.net = net
        self.is_rnn = is_rnn
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, **kwargs):
        x = nn.LayerNorm(inputs.shape[1:])(inputs)
        if self.is_rnn:
            x, _ = self.net(x, **kwargs)
        else:
            x = self.net(x, **kwargs)
        x = inputs + self.dropout(x)
        return x
