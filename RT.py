# Pytorch implementation of RNN-enhanced Transformer

import torch
import torch.nn as nn

from .utils import rnn
from .utils import attention
from .utils import feedforward
from .utils import norm


class RT(nn.Module):
    def __init__(
        self,
        d_model,
        rnn_type='gru',
        kernel=3,
        rnn_layer=1,
        rnn_hsize=256,
        ffn_hsize=256,
        num_heads=4,
        dropout=0.2
    ):
        """RNN enhanced Transformer Block.

        Args:
            d_model (int): Embedded dimension of input.
            rnn_type (str): Rnn type. 'gru', 'lstm', 'simple' is supported.
            kernel (int) : Kernel size of rnn. Batch size for rnn input,
                for easily understanding.
            rnn_layer (int): Number of rnn layers.
            rnn_hsize (int): Hidden size of rnn layer.
            ffn_hsize (int): Hidden size of ffn.
            num_heads (int): Number of heads in Multi-head attention.
            dropout (float): Dropout rate.

        """
        super(RT, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rnn = rnn.RNN(d_model, rnn_type, kernel, rnn_layer,
            rnn_hsize, dropout, self.device)
        mha = attention.MHA(d_model, num_heads, dropout)
        ffn = feedforward.FFN(d_model, ffn_hsize, dropout)

        self.mha_norm = norm.ResidualNorm(mha, d_model)
        self.ffn_norm = norm.ResidualNorm(ffn, d_model)

    def forward(self, inputs, mask=None):
        """Forward propagation of RT.

        Args:
            inputs (torch.Tensor): Input tensor. Shape is [B, L, D]
            mask (torch.Tensor): Mask for attention. If None, calculation
                of masking will not computed.

        Returns:
            torch.Tensor

        """
        x = self.rnn(inputs)
        x = self.mha_norm(x, mask=mask)
        x = self.ffn_norm(x)
        return x

def get_rtrans(d_model, **kwargs):
    return RT(d_model, **kwargs)
