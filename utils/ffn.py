# This is the pytorch implementation of feed forward layer

import torch
import torch.nn as nn


class FFN(nn.Module):
    def __init__(self, d_model, hsize, dropout):
        self.layer_1 = nn.Linear(d_model, hsize)
        self.layer_2 = nn.Linear(hsize, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.dropout(x)
        return x
