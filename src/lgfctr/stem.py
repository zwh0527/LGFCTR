# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 20:11:30 2023

@author: knight
"""

import torch.nn as nn


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv5x5(in_planes, out_planes, stride=1):
    """5x5 convolution with padding"""
    #return nn.Sequential(conv3x3(in_planes, out_planes),
    #                     conv3x3(out_planes, out_planes))
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride, padding=2, bias=False)


def conv7x7(in_planes, out_planes, stride=1):
    """7x7 convolution with padding"""
    #return nn.Sequential(conv5x5(in_planes, out_planes),
    #                     conv3x3(out_planes, out_planes))
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride, padding=3, bias=False)


class CNNBlock1x1(nn.Module):
    def __init__(self, in_planes, planes):
        super().__init__()
        self.conv1 = conv1x1(in_planes, in_planes)
        self.conv2 = conv1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.conv2(self.relu(self.bn1(self.conv1(x))))


class CNNBlock3x3(nn.Module):
    def __init__(self, in_planes, planes):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(True)  # 原来是LeakyReLU

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.conv2(self.relu(self.bn1(self.conv1(x))))


class ResBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential(
            conv1x1(in_planes, planes, stride=stride),
            nn.BatchNorm2d(planes)
        )
        self.shortcut = None
        if stride > 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, planes, stride=stride),
                nn.BatchNorm2d(planes)
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = x
        y = self.relu(self.bn1(self.conv1(y)))
        y = self.bn2(self.conv2(y))

        if self.shortcut is not None:
            x = self.shortcut(x)

        return self.relu(x + y)


class Stem(nn.Module):
    """
    shallow feature extraction stem
    """

    def __init__(self, config):
        super(Stem, self).__init__()

        self.config = config
        self.layers = []
        in_channels = 1
        for channels in config['STEM_DIMS']:
            layer = ResBlock(in_channels, channels)
            in_channels = channels
            self.layers.append(layer)
        for idx, channels in enumerate(config['STEM_DIMS2']):
            stride = 2 if idx == 0 else 1
            layer = ResBlock(in_channels, channels, stride)
            in_channels = channels
            self.layers.append(layer)
        self.layers = nn.Sequential(*tuple(self.layers))

    def forward(self, data):
        return self.layers(data)
