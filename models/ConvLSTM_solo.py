import torch
import logging
import sys
sys.path.append('..')
from torch import nn
from net_parameter import cnn_encoder_params,cnn_decoder_params,convlstm_encoder,convlstm_decoder
from collections import OrderedDict

def make_layers(block):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                    padding=v[2])
            layers.append((layer_name, layer))
        elif 'dropout' in layer_name:
            layer = nn.Dropout2d(p=v[0])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            transposeConv2d = nn.ConvTranspose2d(in_channels=v[0], out_channels=v[1],
                                                 kernel_size=v[2], stride=v[3],
                                                 padding=v[4])
            layers.append((layer_name, transposeConv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif 'conv' in layer_name:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                               kernel_size=v[2], stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        else:
            raise NotImplementedError

    return nn.Sequential(OrderedDict(layers))


class Encoder(nn.Module):
    def __init__(self, rnns):
        super().__init__()
        self.blocks = len(rnns)

        for index, (rnn) in enumerate(rnns, 1):
            # index sign from 1
            setattr(self, 'rnn' + str(index), rnn)

    def forward_by_stage(self, inputs, rnn):
        print('encoder shape:',inputs.shape)  # [52,4,16,20,100] ; batch size=4
        outputs_stage, state_stage = rnn(inputs, None)
        return outputs_stage, state_stage

    def forward(self, inputs):
        inputs = inputs.transpose(0, 1)  # to 52,B,1,20,100
        hidden_states = []
        logging.debug(inputs.size())
        for i in range(1, self.blocks + 1):
            inputs, state_stage = self.forward_by_stage(
                inputs,getattr(self, 'rnn' + str(i)))
            hidden_states.append(state_stage)
        return tuple(hidden_states)


class Forecaster(nn.Module):
    def __init__(self, rnns):
        super().__init__()

        self.blocks = len(rnns)

        for index, (rnn) in enumerate(rnns):
            setattr(self, 'rnn' + str(self.blocks-index), rnn)

    def forward_by_stage(self, input, state, rnn):
        input, state_stage = rnn(input, state, seq_len=12)
        print('decoder shape:', input.shape)
        return input

        # input: 5D S*B*I*H*W

    def forward(self, hidden_states):
        input = self.forward_by_stage(None, hidden_states[-1], getattr(self, 'rnn2'))
        for i in list(range(1, self.blocks))[::-1]:
            input = self.forward_by_stage(input, hidden_states[i-1], getattr(self, 'rnn' + str(i)))
        input = input.permute(1, 0, 2, 3, 4)
        return input

class EF(nn.Module):

    def __init__(self, encoder, forecaster):
        super().__init__()
        self.encoder = encoder
        self.forecaster = forecaster
        self.MSE_criterion = torch.nn.MSELoss()

    def forward(self, input, labels):
        state = self.encoder(input)
        output = self.forecaster(state)
        loss = self.MSE_criterion(output, labels)
        return output, loss

class EF_Conv(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.net = args
        self.MSE_criterion = torch.nn.MSELoss()
        if self.net == 'convlstm':
            self.encoder_params = convlstm_encoder
            self.decoder_params = convlstm_decoder
        elif self.net == 'cnn':
            self.encoder_params = cnn_encoder_params
            self.decoder_params = cnn_decoder_params
        self.encoder = Encoder(self.encoder_params[0])
        self.forecaster = Forecaster(self.decoder_params[0])
        self.conv_last = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64,
                            out_channels=1,
                            kernel_size=1,
                            stride=1,
                            padding=0)
        )

    def forward(self, inputs, labels):
        state = self.encoder(inputs)
        output = self.forecaster(state)
        batch_size, seq_number, input_channel, height, width = output.size()  # [52,4,1,20,100]
        output = torch.reshape(output, (-1, input_channel, height, width))  # [208,1,20,100]
        output = self.conv_last(output)
        output = torch.reshape(output, (batch_size, seq_number, output.size(1),
                              output.size(2), output.size(3)))
        loss = self.MSE_criterion(output, labels)
        return output, loss


from torch.autograd import Variable
# train_X = torch.load('C:\\Users\\123\\Desktop\\github\\PycharmProjects\\ENSO_zxy\\DataProcess\\train_X_CMIP.pt')
# train_Y = torch.load('C:\\Users\\123\\Desktop\\github\\PycharmProjects\\ENSO_zxy\\DataProcess\\train_Y_CMIP.pt')
# train_X = train_X.to(torch.float32)
# train_Y = train_Y.to(torch.float32)
# train_tensor = TensorDataset(train_X, train_Y)
# train_loader = torch.utils.data.DataLoader(train_tensor,
#                                                batch_size=8,
# 
#                                                shuffle=False)
x = torch.rand((2, 12, 1, 20, 50))
y = torch.rand((2, 12, 1, 20, 50))
net=EF_Conv('convlstm')
output, loss = net(x,y)
print(loss)
print(net)
# optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
# for epoch in range(2):
#     net.train()
#     train_loss = 0.0
#     train_total = 0.0
#     valid_losses = []
#     for i, data in enumerate(train_loader, 0):
#         length = len(train_loader)  # length = 47500 / batch_size
#         _, inputs, labels = data
#         inputs, labels = Variable(inputs), Variable(labels)
#         optimizer.zero_grad()
#         outputs, loss = net(inputs, labels)
#         loss.backward()
#         optimizer.step()
