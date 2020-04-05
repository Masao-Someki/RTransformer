# pytorch implementation of custom rnn.
import torch
import torch.nn as nn

from . import norm


class RNN(nn.Module):
    def __init__(self, d_model, rnn_type, kernel, nlayer,
        hsize, dropout, device):
        super(RNN, self).__init__()
        self.kernel = kernel

        if rnn_type == 'gru':
            net = nn.GRU(
                input_size=d_model,
                hidden_size=hsize,
                num_layers=nlayer,
                bias=True,
                batch_first=False,
                dropout=dropout,
                bidirectional=False
            )
        elif rnn_type == 'lstm':
            net = nn.LSTM(
                input_size=d_model,
                hidden_size=hsize,
                num_layers=nlayer,
                bias=True,
                batch_first=False,
                dropout=dropout,
                bidirectional=False
            )
        elif rnn_type == 'simple':
            net = nn.RNN(
                input_size=d_model,
                hidden_size=hsize,
                num_layers=nlayer,
                nonlinearity='tanh',
                bias=True,
                batch_first=False,
                dropout=dropout,
                bidirectional=False
            )
        self.rnn = norm.ResidualNorm(net, d_model, dropout, is_rnn=True)
<<<<<<< HEAD
        self.device = device
=======
>>>>>>> e3365d021573f413a4a2cb736f9c28beafe507e0

    def forward(self, inputs):
        b, l, d = inputs.shape
        x = self.set_input(inputs)
        x = self.rnn(x)
        x = x[:, -1, :]
        return x.view(b, l, d)

    def set_input(self, x):
        """set kernel_size as input batch size."""
        b, l, d = x.shape
        x = self.pad(x)
        index = [id for j in range(l + self.kernel - 2)
                    for id in range(j, j + self.kernel)]
<<<<<<< HEAD
        index = torch.LongTensor(index).to(self.device)
=======
        index = torch.LongTensor(index).cuda()
>>>>>>> e3365d021573f413a4a2cb736f9c28beafe507e0
        x = torch.index_select(x, 1, index[:int(self.kernel*l)])
        x = x.reshape(b, l, self.kernel, -1)
        d = x.shape[-1]
        return x.view(-1, self.kernel, d)

    def pad(self, inputs):
        b, l, d = inputs.shape
<<<<<<< HEAD
        zeros = torch.zeros((self.kernel-1, d)).to(self.device)
=======
        zeros = torch.zeros((self.kernel-1, d)).cuda()
>>>>>>> e3365d021573f413a4a2cb736f9c28beafe507e0
        zeros = zeros.unsqueeze(0).repeat(b, 1, 1)
        return torch.cat((zeros, inputs), dim=1)
