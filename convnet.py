import torch
import torch.nn as nn
import torch.nn.functional as F


class IntelBlock(nn.Module):
    def __init__(self, in_channels, out_сhannels, downsample=False):
        '''
        in_channels: int,
            Number of channels in the input.
        out_channels: int,
            Number of channels in the output.
        downsample: bool,
            True if in_channels != out_channels.
        '''
        super(IntelBlock, self).__init__()
        stride, dilation, padding = 1, 1, 1

        self.module = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_сhannels,
                kernel_size=3,
                stride=stride,
                dilation=dilation,
                padding=padding
            ),
            nn.ReLU(),
            nn.BatchNorm2d(out_сhannels),
            nn.MaxPool2d(2, 2)
        )
        self.relu = nn.ReLU()

        self.downsample = nn.Identity()
        if downsample:
            self.downsample = nn.Conv2d(
                in_channels,
                out_сhannels,
                kernel_size=1,
                stride=stride
            )

    def forward(self, input):
        output = self.module(input)
        output = self.relu(output)

        return output


class IntelNet(nn.Module):
    def __init__(self, out_dim=10, in_channels=3):
        '''
        out_dim: int,
            Number of classes in the target dataset. By default 6 for Intel dataset.
        in_channels: int,
            Number of channels in the input image. By default 3.
        '''
        super(IntelNet, self).__init__()

        depth_list = [32, 64, 128]
        self.out_shape = depth_list[-1]

        self.extract_layer = nn.Conv2d(
            3,
            depth_list[0],
            kernel_size=3,
            stride=1,
            dilation=1
        )
        self.blocks = nn.Sequential(
            IntelBlock(depth_list[0], depth_list[0]),
            IntelBlock(depth_list[0], depth_list[1], downsample=True),
            IntelBlock(depth_list[1], depth_list[1]),
            IntelBlock(depth_list[1], depth_list[2], downsample=True),
            IntelBlock(depth_list[2], depth_list[2]),

        )

        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.3)
        self.linear = nn.Linear(depth_list[-1], out_dim)

    def forward(self, x):
        output = self.extract_layer(x)

        output = self.blocks(output)

        output = self.dropout(self.pooling(output))

        return self.linear(output.reshape(-1, self.out_shape))
