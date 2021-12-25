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
            nn.Conv2d(
                out_сhannels, 
                out_сhannels, 
                kernel_size=1, 
                stride=stride
            ),
            nn.BatchNorm2d(out_сhannels)
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
        residual = input

        output = self.module(input)
        output = self.relu(output + self.downsample(residual))

        return output
        

class IntelNet(nn.Module):
    def __init__(self, out_dim=6, in_channels=3):
        '''
        out_dim: int,
            Number of classes in the target dataset. By default 6 for Intel dataset.
        in_channels: int,
            Number of channels in the input image. By default 3.
        '''
        super(IntelNet, self).__init__()

        depth_list = [32, 64, 128]
        self.out_shape = depth_list[-1]

        self.extract_layer =  nn.Conv2d(
            3, 
            depth_list[0], 
            kernel_size=7, 
            stride=3, 
            dilation=1
        )
        
        self.block1 = IntelBlock(depth_list[0], depth_list[0])
        self.block2 = IntelBlock(depth_list[0], depth_list[1], downsample=True)
        self.block3 = IntelBlock(depth_list[1], depth_list[1])
        self.block4 = IntelBlock(depth_list[1], depth_list[2], downsample=True)

        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(depth_list[-1], out_dim)

    def forward(self, x):

        output = self.extract_layer(x)

        output = self.block1(output)
        output = self.block2(output)
        output = self.block3(output)
        output = self.block4(output)

        output = self.pooling(output)

        return self.linear(output.reshape(-1, self.out_shape))