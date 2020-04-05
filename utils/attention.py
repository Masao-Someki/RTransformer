# This is an implementation of multi-head attention layer
import math
import sys

import torch
import torch.nn as nn

class MHA(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super(MHA, self).__init__()
        self.heads = nhead
        self.d_k = d_model // nhead
        self.linear_1 = Linear(d_model, d_model*3)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = Linear(d_model, d_model)

    def forward(self, inputs, mask=None):
        """Forward propagation.

        Args:
            inputs: (B, L, D)
            mask: (B, L, L)
        """
        b, l, d = inputs.shape
        x = self.linear_1(inputs)
        q, k, v = map(self.split_heads, torch.split(x, d, dim=2))
        k = k.transpose(2, 3) # (b, h, l, d_k)
        scores = torch.matmul(q, k) # (b, h, l, l)
        scores = scores / math.sqrt(self.d_k) # (b, h, l, l)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)
            min_value = float(sys.float_info.min)
            scores = scores.masked_fill(mask, min_value)
            attn = torch.softmax(scores, dim=-1)
            attn = attn.masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)
        x = self.dropout(attn)
        x = torch.matmul(x, v) # (b, h, l, d_k)
        x = x.transpose(1, 2).contiguous() # (b, l, h, d_k)
        x = x.view(b, l, d)
        x = self.linear_2(x)
        return x

    def split_heads(self, x):
        b, l, d = x.shape
        return x.reshape((b, l, self.heads, d // self.heads)).transpose(1, 2)

class Linear(nn.Module):
    def __init__(self, din, dout):
        super(Linear, self).__init__()
        self.din = din
        self.layer = nn.Linear(din, dout)

    def forward(self, inputs):
        b, l, _ = inputs.shape
        x = inputs.reshape(-1, self.din)
        x = self.layer(x)
        x = x.reshape(b, l, -1)
        return x
