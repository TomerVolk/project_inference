import torch
import torch.nn as nn


class InceptionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, max_pool_kernel=2):
        super(InceptionLayer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, padding=3)
        self.conv5 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=21, padding=10)
        self.conv7 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=35, padding=17)
        self.max_pool = nn.MaxPool1d(kernel_size=max_pool_kernel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        x = torch.cat((x1, x3, x5, x7), dim=1)
        return self.relu(self.max_pool(x))


class InceptionNet(nn.Module):
    def __init__(self, in_dim=21, linear_out=500, net_out=2):
        super(InceptionNet, self).__init__()
        self.inception_seq = nn.Sequential(
            InceptionLayer(in_channels=1, out_channels=1),
            InceptionLayer(in_channels=4, out_channels=2),
            InceptionLayer(in_channels=8, out_channels=6),
            InceptionLayer(in_channels=24, out_channels=20)
        )
        self.flatten = nn.Flatten()
        stupid = torch.ones([1, 1, in_dim])
        stupid = self.inception_seq(stupid)
        stupid = self.flatten(stupid)
        linear_in_dim = stupid.shape[-1]

        self.linear_head = nn.Sequential(
            nn.Linear(in_features=linear_in_dim, out_features=linear_out),
            nn.ReLU(),
            nn.Linear(in_features=linear_out, out_features=200),
            nn.ReLU(),
            nn.Linear(in_features=200, out_features=net_out)
        )

    def forward(self, x):
        x = self.inception_seq(x)
        x = self.flatten(x)
        return self.linear_head(x)
