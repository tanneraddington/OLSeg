import pytorch_utils
import torch.nn as nn

class TopDownLayer(nn.Module):
    '''
    This class represents a topdown convolutional layer that is used in RPN
    '''
    def __init__(self, in_channels, out_channels):
        super(TopDownLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1)
        self.padding = pytorch_utils.SamePad2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d()