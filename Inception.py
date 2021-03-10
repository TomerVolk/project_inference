import torch
import torch.nn as nn


class InceptionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, max_pool_kernel=2):
        super(InceptionLayer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, padding=2)
        self.max_pool = nn.MaxPool1d(kernel_size=max_pool_kernel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x = torch.cat((x1, x3, x5), dim=1)
        return self.relu(self.max_pool(x))


class InceptionNet(nn.Module):
    def __init__(self, in_dim=21, linear_out=500, net_out=2):
        super(InceptionNet, self).__init__()
        self.inception_seq = nn.Sequential(
            InceptionLayer(in_channels=1, out_channels=1),
            InceptionLayer(in_channels=3, out_channels=4),
            InceptionLayer(in_channels=12, out_channels=20),
            InceptionLayer(in_channels=60, out_channels=70)
        )
        self.flatten = nn.Flatten()
        stupid = torch.ones([1, 1, in_dim])
        stupid = self.inception_seq(stupid)
        stupid = self.flatten(stupid)
        linear_in_dim = stupid.shape[-1]

        self.linear_head = nn.Sequential(
            nn.Linear(in_features=linear_in_dim, out_features=linear_out),
            nn.ReLU(),
            nn.Linear(in_features=linear_out, out_features=net_out)
        )

    def forward(self, x):
        x = self.inception_seq(x)
        x = self.flatten(x)
        return self.linear_head(x)
