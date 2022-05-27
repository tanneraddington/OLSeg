import datetime
import math
import os
import random
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

import pytorch_utils

import utils

class Bottleneck(nn.Module):
    """
    This is the bottleneck class. Used as teh block for resnet.
    """
    # **** check to see what this is
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        Initialize the layers for the NN
        :param inplanes: in channels
        :param planes: out channels
        :param stride:
        :param downsample:
        """
        super(Bottleneck, self).__init__()
        global expansion

        self.downsample = downsample
        self.stride = stride

        # set up the NN layers
        self.conv1 = nn.Conv2d(inplanes,planes,kernel_size=1,stride=stride)
        # planes = number of features.
        self.b_norm1 = nn.BatchNorm2d(planes, eps=0.001, momentum=0.01)
        # from pytorch utils
        self.padding2 = pytorch_utils.SamePad2d(kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(planes, planes,kernel_size=3)
        self.b_norm2 = nn.BatchNorm2d(planes, eps=0.001, momentum=0.01)
        self.conv3 = nn.Conv2d(planes, planes * expansion ,kernel_size=3)
        self.b_norm3 = nn.BatchNorm2d(planes * expansion, eps=0.001, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, res):
        """
        This is the forward method in bottleneck. It will activate layers

        :param res:
        :return:
        """
        residual = res
        # first layer
        output = self.conv1(res)
        output = self.b_norm1(output)
        output = self.relu(output)
        # second layer
        output = self.padding2(output)
        output = self.conv2(output)
        output = self.b_norm2(output)
        output = self.relu(output)
        # 3rd (final) layer
        output = self.conv3(output)
        output = self.b_norm3(output)

        if self.downsample is not None:
            residual = self.downsample(res)
        output += residual
        output = self.relu(output)
        return output

class ResNet(nn.Module):
    """
    This is the resnet class. It wraps the layers defined in Bottleneck
    """
    def __init__(self, arch, stage5=False):
        super(ResNet,self).__init__()
        assert arch in ["resnet50", "resnet101"]
        self.inplanes = 64
        # determine the number of layers for resnet
        l3 = {"resnet50": 6, "resnet101": 23}[arch]
        self.layers = [3,4,l3,3]
        self.block = Bottleneck
