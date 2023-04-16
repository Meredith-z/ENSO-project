import torch


class CNNnet(torch.nn.Module):
    def __init__(self):
        super(CNNnet, self).__init__()
        self.MSE_criterion = torch.nn.MSELoss()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=16,
                            kernel_size=3,
                            stride=1,
                            padding=1),

            torch.nn.ReLU(),
            torch.nn.Dropout2d(0.2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 3, 1, 1),

            torch.nn.ReLU(),
            torch.nn.Dropout2d(0.2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 16, 3, 1, 1),

            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 1, 1, 1, 0),

            torch.nn.ReLU()
        )

    def forward(self, x, y):
        batch_size, seq_number, input_channel, height, width = x.size()  # [52,4,1,20,100]
        x = torch.reshape(x, (-1, input_channel, height, width))  # [208,1,20,100]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = torch.reshape(x, (batch_size, seq_number, x.size(1),
                              x.size(2), x.size(3)))
        loss = self.MSE_criterion(x, y)
        return x, loss

