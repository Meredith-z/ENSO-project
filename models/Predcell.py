__author__ = 'yunbo'

import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PredCell(nn.Module):
    def __init__(self, shape, input_channels, filter_size, num_features,layer_norm=1):
        super(PredCell, self).__init__()

        self.height = shape[0]  # H, W
        self.width = shape[1]
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features
        # in this way the output has the same size
        self.padding = (filter_size - 1) // 2
        self._forget_bias = 1.0
        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(self.input_channels, self.num_features * 7, kernel_size=filter_size, stride=1, padding=self.padding, bias=False),
                nn.LayerNorm([self.num_features * 7, self.height, self.width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(self.num_features, self.num_features * 4, kernel_size=filter_size, stride=1, padding=self.padding, bias=False),
                nn.LayerNorm([self.num_features * 4, self.height, self.width])
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(self.num_features, self.num_features * 3, kernel_size=filter_size, stride=1, padding=self.padding, bias=False),
                nn.LayerNorm([self.num_features * 3, self.height, self.width])
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(self.num_features * 2, self.num_features, kernel_size=filter_size, stride=1, padding=self.padding, bias=False),
                nn.LayerNorm([self.num_features, self.height, self.width])
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(self.input_channels, self.num_features * 7, kernel_size=filter_size, stride=1, padding=self.padding, bias=False),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(self.num_features, self.num_features * 4, kernel_size=filter_size, stride=1, padding=self.padding, bias=False),
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(self.num_features, self.num_features * 3, kernel_size=filter_size, stride=1, padding=self.padding, bias=False),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(self.num_features * 2, self.num_features, kernel_size=filter_size, stride=1, padding=self.padding, bias=False),
            )
        self.conv_last = nn.Conv2d(self.num_features * 2, self.num_features, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, inputs=None, hidden_state=None, seq_len=12):
        
        if hidden_state is None:
            h_t = torch.zeros(inputs.size(1), self.num_features, self.height,
                             self.width).to(device)
            c_t = torch.zeros(inputs.size(1), self.num_features, self.height,
                             self.width).to(device)
            m_t = torch.zeros(inputs.size(1), self.num_features, self.height,
                             self.width).to(device)
        else:
            h_t, c_t, m_t = hidden_state

        output_inner = []

        for t in range(seq_len):
            if inputs is None:
                x_t = torch.zeros(h_t.size(0), self.input_channels, self.height,
                                self.width).to(device)
            else:
                x_t = inputs[t,...]
            x_concat = self.conv_x(x_t)
            h_concat = self.conv_h(h_t)
            m_concat = self.conv_m(m_t)
            i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_features, dim=1)
            i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_features, dim=1)
            i_m, f_m, g_m = torch.split(m_concat, self.num_features, dim=1)

            i_t = torch.sigmoid(i_x + i_h)
            f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
            g_t = torch.tanh(g_x + g_h)

            c_new = f_t * c_t + i_t * g_t

            i_t_prime = torch.sigmoid(i_x_prime + i_m)
            f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
            g_t_prime = torch.tanh(g_x_prime + g_m)

            m_new = f_t_prime * m_t + i_t_prime * g_t_prime

            mem = torch.cat((c_new, m_new), 1)
            o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
            h_new = o_t * torch.tanh(self.conv_last(mem))
            output_inner.append(h_new)

        return torch.stack(output_inner), (h_new, c_new, m_new)









