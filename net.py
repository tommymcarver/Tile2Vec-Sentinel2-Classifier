import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import time

from classifierLoading import tile_dataloader

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Net(nn.Module):
    def __init__(self, num_blocks, in_channels=4, z_dim=512):
        super(Net, self).__init__()
        self.in_channels = in_channels
        self.z_dim = z_dim
        #creates scaled-down numbers to use as layer dimensions
        self.z_dim2=round(z_dim/2)
        self.z_dim4=round(z_dim/4)
        self.z_dim8=round(z_dim/8)

        self.in_planes = self.z_dim8

        self.conv1 = nn.Conv2d(self.in_channels, self.z_dim8, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.z_dim8)
        self.layer1 = self._make_layer(self.z_dim8, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(self.z_dim4, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(self.z_dim2, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(self.z_dim, num_blocks[3], stride=2)
        self.layer5 = self._make_layer(self.z_dim, num_blocks[4],
            stride=2)

        self.layer6 = self._make_layer(2, 1, stride=1)

        self.sf = nn.Sequential(nn.Softmax(dim=1))

    def _make_layer(self, planes, num_blocks, stride, no_relu=False):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_planes, planes, stride=stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def encode(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.sf(x)
        x = F.avg_pool2d(x, 4)
        z = x.view(x.size(0), -1)
        return z

    def forward(self, x):
        return self.encode(x)

    def loss(self):
        return nn.CrossEntropyLoss()
