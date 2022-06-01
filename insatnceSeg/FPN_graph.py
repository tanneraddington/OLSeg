import pytorch_utils
import torch.nn as nn
import torch.nn.functional as F

class TopDownLayer(nn.Module):
    '''
    This class represents a topdown convolutional layer that is used in RPN
    '''
    def __init__(self, in_channels, out_channels):
        '''

        :param in_channels:
        :param out_channels:
        '''
        super(TopDownLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1)
        self.padding = pytorch_utils.SamePad2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1)

    def forward(self,x ,y):
        '''
        This is the forward method, it uses the specific layers in
        the constructor
        :param x:
        :param y:
        :return:
        '''
        y = F.upsample(y, scale_factor=2)
        x = self.conv1(x)
        return self.conv2(self.padding(x+y))

class FPN(nn.Module):
    '''
    FPN is used in the paper as a first stage network which in parallel predicts class
    and box offset
    '''
    def __init__(self, C1, C2, C3, C4, C5, out_channels):
        '''
        Construct the final layers used for forward
        :param C1:
        :param C2:
        :param C3:
        :param C4:
        :param C5:
        :param out_channels:
        '''
        super(FPN, self).__init__()
        self.out_channels = out_channels
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.C4 = C4
        self.C5 = C5
        # Create the P layers
        self.P6 = nn.MaxPool2d(kernel_size=1, stride=2)
        self.P5_conv1 = nn.Conv2d(2048, self.out_channels, kernel_size=1, stride=1)
        self.P5_conv2 = nn.Sequential(
            pytorch_utils.SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )
        self.P4_conv1 = nn.Conv2d(1024, self.out_channels, kernel_size=1, stride=1)
        self.P4_conv2 = nn.Sequential(
            pytorch_utils.SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )
        self.P3_conv1 = nn.Conv2d(512, self.out_channels, kernel_size=1, stride=1)
        self.P3_conv2 = nn.Sequential(
            pytorch_utils.SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )
        self.P2_conv1 = nn.Conv2d(256, self.out_channels, kernel_size=1, stride=1)
        self.P2_conv2 = nn.Sequential(
            pytorch_utils.SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )

    def forward(self,x):
        '''
        This forward method uses saved previous outputs to run
        through all the convolutional layers
        :param x:
        :return:
        '''
        x = self.C1(x)
        x = self.C2(x)
        c2_out = x
        x = self.C3(x)
        c3_out = x
        x = self.C4(x)
        c4_out = x
        x = self.C5(x)
        # use the outputs to generate p outs
        p5_out = self.P5_conv1(x)
        p4_out = self.P4_conv1(c4_out) + F.upsample(p5_out, scale_factor=2)
        p3_out = self.P3_conv1(c3_out) + F.upsample(p4_out, scale_factor=2)
        p2_out = self.P2_conv1(c2_out) + F.upsample(p3_out, scale_factor=2)

        p5_out = self.P5_conv2(p5_out)
        p4_out = self.P4_conv2(p4_out)
        p3_out = self.P3_conv2(p3_out)
        p2_out = self.P2_conv2(p2_out)
        # this is used as an anchor for RPN
        p6_out = self.P6(p5_out)
        return [p2_out, p3_out, p4_out, p5_out, p6_out]


