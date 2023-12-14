"""
Adapted from:
https://github.com/zhiyongc/GRU-D/blob/master/GRUD.py

Citation:
@article{che2018recurrent,
  title={Recurrent neural networks for multivariate time series with missing values},
  author={Che, Zhengping and Purushotham, Sanjay and Cho, Kyunghyun and Sontag, David and Liu, Yan},
  journal={Scientific reports},
  volume={8},
  number={1},
  pages={6085},
  year={2018},
  publisher={Nature Publishing Group}
}
"""
# mypy: ignore-errors

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter


class FilterLinear(nn.Module):
    def __init__(self, in_features, out_features, filter_square_matrix, device, bias=True):
        """
        filter_square_matrix : filter square matrix, whose each elements is 0 or 1.
        """
        super(FilterLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.filter_square_matrix = None
        self.filter_square_matrix = Variable(filter_square_matrix, requires_grad=False).to(device)
        self.weight = Parameter(torch.Tensor(out_features, in_features).to(device))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features).to(device))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return F.linear(  # pylint: disable=not-callable
            x,
            self.filter_square_matrix.mul(self.weight),  # type: ignore
            self.bias,
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "in_features="
            + str(self.in_features)
            + ", out_features="
            + str(self.out_features)
            + ", bias="
            + str(self.bias is not None)
            + ")"
        )


class GRUD(nn.Module):
    def __init__(self, input_size, hidden_size, device, output_last=False):
        """
        Recurrent Neural Networks for Multivariate Times Series with Missing Values
        GRU-D: GRU exploit two representations of informative missingness patterns, i.e., masking and time interval.
        Implemented based on the paper:
        @article{che2018recurrent,
          title={Recurrent neural networks for multivariate time series with missing values},
          author={Che, Zhengping and Purushotham, Sanjay and Cho, Kyunghyun and Sontag, David and Liu, Yan},
          journal={Scientific reports},
          volume={8},
          number={1},
          pages={6085},
          year={2018},
          publisher={Nature Publishing Group}
        }

        GRU-D:
            input_size: variable dimension of each time
            hidden_size: dimension of hidden_state
            mask_size: dimension of masking vector
        """

        super(GRUD, self).__init__()

        self.hidden_size = hidden_size
        self.delta_size = input_size
        self.mask_size = input_size
        self.device = device

        self.identity = torch.eye(input_size).to(device)
        self.zeros = Variable(torch.zeros(input_size).to(device))
        self.zeros_hidden = Variable(torch.zeros(hidden_size).to(device))

        self.zl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size).to(device)
        self.rl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size).to(device)
        self.hl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size).to(device)

        self.gamma_x_l = FilterLinear(self.delta_size, self.delta_size, self.identity, device=device, bias=True)

        self.gamma_h_l = nn.Linear(self.delta_size, self.hidden_size).to(device)

        self.output_last = output_last

    def step(self, x, x_last_obsv, h, mask, delta):
        batch_size = x.shape[0]  # noqa  # pylint: disable=unused-variable
        dim_size = x.shape[1]  # noqa  # pylint: disable=unused-variable

        delta_x = torch.exp(-torch.max(self.zeros, self.gamma_x_l(delta)))
        delta_h = torch.exp(-torch.max(self.zeros_hidden, self.gamma_h_l(delta)))  # pylint: disable=not-callable

        # print('mask', mask.shape)
        # print('x', x.shape)
        # print('delta_x', delta_x.shape)
        # print('x_last_obsv', x_last_obsv.shape)

        x = mask * x + (1 - mask) * (delta_x * x_last_obsv)
        # print('h', h.shape)
        # print('delta_h', delta_h.shape)
        h = delta_h * h

        combined = torch.cat((x, h, mask), 1)
        z = torch.sigmoid(self.zl(combined))  # pylint: disable=not-callable
        r = torch.sigmoid(self.rl(combined))  # pylint: disable=not-callable
        combined_r = torch.cat((x, r * h, mask), 1)
        h_tilde = torch.tanh(self.hl(combined_r))  # pylint: disable=not-callable
        h = (1 - z) * h + z * h_tilde

        return h

    def forward(self, input):  # pylint: disable=redefined-builtin
        batch_size = input.size(0)
        type_size = input.size(1)  # noqa  # pylint: disable=unused-variable
        step_size = input.size(2)
        spatial_size = input.size(3)  # noqa  # pylint: disable=unused-variable

        Hidden_State = self.initHidden(batch_size)
        X = torch.squeeze(input[:, 0, :, :])
        X_last_obsv = torch.squeeze(input[:, 1, :, :])
        Mask = torch.squeeze(input[:, 2, :, :])
        Delta = torch.squeeze(input[:, 3, :, :])

        outputs = None
        for i in range(step_size):
            Hidden_State = self.step(
                torch.squeeze(X[:, i : i + 1, :]),
                torch.squeeze(X_last_obsv[:, i : i + 1, :]),
                Hidden_State,
                torch.squeeze(Mask[:, i : i + 1, :]),
                torch.squeeze(Delta[:, i : i + 1, :]),
            )
            if outputs is None:
                outputs = Hidden_State.unsqueeze(1)
            else:
                outputs = torch.cat((outputs, Hidden_State.unsqueeze(1)), 1)

        if self.output_last:
            return outputs[:, -1, :]  # type: ignore
        else:
            return outputs

    def initHidden(self, batch_size):
        Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size).to(self.device))
        return Hidden_State

    def get_input_for_grud(self, t, y, y_mask):
        # T B D
        t = t[:, :, None, :]
        y = y[:, :, None, :]
        y_mask = y_mask[:, :, None, :]

        t_delta = t[1:] - t[:-1]
        t_delta_mat = torch.cat((torch.zeros_like(t_delta[0:1, ...]), t_delta), dim=0)
        last_y_mat = torch.cat((y[0:1, ...], y[:-1, ...]), dim=0)

        for i in range(1, t_delta_mat.shape[0]):
            last_mask = y_mask[i - 1, ...].to(last_y_mat)
            last_y_mat[i, ...] = last_y_mat[i, ...] + last_y_mat[i - 1, ...] * (1 - last_mask)

        gru_d_input = torch.cat((y, last_y_mat, y_mask.to(y), t_delta_mat), dim=2).permute((1, 2, 0, 3)).to(self.device)
        return gru_d_input
