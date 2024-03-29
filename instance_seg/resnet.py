import torch.nn as nn

import pytorch_utils

class Bottleneck(nn.Module):
    """
    This is the bottleneck class. Used as the block for resnet.
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

class Resnet(nn.Module):
    """
    This is the resnet class. It wraps the layers defined in Bottleneck. The goal of
    resnet is to find the regions of interest
    """
    def __init__(self, arch, stage5=False):
        super(Resnet,self).__init__()
        assert arch in ["resnet50", "resnet101"]
        self.inplanes = 64
        # determine the number of layers for resnet
        l3 = {"resnet50": 6, "resnet101": 23}[arch]
        self.layers = [3,4,l3,3]
        self.block = Bottleneck
        self.stage5 = stage5

        #create the sequential layers
        conv_seq = nn.Conv2d(3,64,kernel_size=7, stride=2, padding=3)
        b_norm_seq = nn.BatchNorm2d(64,eps= 0.001, momentum= 0.01)
        relu_seq = nn.ReLU(inplace=True)
        s_pad_seq = pytorch_utils.SamePad2d(kernel_size=3, stride=2)
        max_pool_seq = nn.MaxPool2d(kernel_size=3, stride=2)
        # Create the 4 or 5 conv layers
        self.C1 = nn.Sequential(conv_seq,b_norm_seq,relu_seq,s_pad_seq,max_pool_seq)
        self.C2 = self.make_layer(self.block, 64, self.layers[0])
        self.C3 = self.make_layer(self.block, 128, self.layers[1], stride=2)
        self.C4 = self.make_layer(self.block, 256, self.layers[2], stride=2)
        if self.stage5:
            self.C5 = self.make_layer(self.block,512,self.layers[3],stride=2)
        else:
            self.C5 = None


    def make_layer(self,block,planes,blocks, stride=1):
        '''
        The make_layer method is a helper method for the constructor. It makes more complex layer combinations
        using the block planes and stride.
        :param block:
        :param planes:
        :param blocks:
        :param stride:
        :return:
        '''
        downsample = None
        # check to see if the stride is not one or if the inplanes are not plane * expansion
        if stride != 1 or self.inplanes != planes * block.expansion:
            conv = nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride)
            norm = nn.BatchNorm2d(planes * block.expansion, eps=0.001, momentum=0.01)
            downsample = nn.Sequential(conv, norm)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))


        return nn.Sequential

    def forward(self,x):
        '''
        This is the forward method! It uses the Convolutional layers that were
        generated in bottleneck and in the constructor to run the forward direciton
        of the neural network.
        :param x:
        :return:
        '''
        x = self.C1(x)
        x = self.C2(x)
        x = self.C3(x)
        x = self.C4(x)
        x = self.C5(x)
        return x

    def stages(self):
        return [self.C1, self.C2, self.C3, self.C4, self.C5]

