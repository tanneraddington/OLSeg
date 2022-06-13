import math
import numpy as np
from torch.autograd import Variable
from pytorch_utils import log2
import torch
from torch import nn
from .crop_and_resize import CropAndResizeFunction, CropAndResize
##from roialign.roi_align.crop_and_resize import CropAndResizeFunction

class RoIAlign(nn.Module):

    def __init__(self, crop_height, crop_width, extrapolation_value=0, transform_fpcoor=True):
        super(RoIAlign, self).__init__()

        self.crop_height = crop_height
        self.crop_width = crop_width
        self.extrapolation_value = extrapolation_value
        self.transform_fpcoor = transform_fpcoor

    def forward(self, featuremap, boxes, box_ind):
        """
        RoIAlign based on crop_and_resize.
        See more details on https://github.com/ppwwyyxx/tensorpack/blob/6d5ba6a970710eaaa14b89d24aace179eb8ee1af/examples/FasterRCNN/model.py#L301
        :param featuremap: NxCxHxW
        :param boxes: Mx4 float box with (x1, y1, x2, y2) **without normalization**
        :param box_ind: M
        :return: MxCxoHxoW
        """
        x1, y1, x2, y2 = torch.split(boxes, 1, dim=1)
        image_height, image_width = featuremap.size()[2:4]

        if self.transform_fpcoor:
            spacing_w = (x2 - x1) / float(self.crop_width)
            spacing_h = (y2 - y1) / float(self.crop_height)

            nx0 = (x1 + spacing_w / 2 - 0.5) / float(image_width - 1)
            ny0 = (y1 + spacing_h / 2 - 0.5) / float(image_height - 1)
            nw = spacing_w * float(self.crop_width - 1) / float(image_width - 1)
            nh = spacing_h * float(self.crop_height - 1) / float(image_height - 1)

            boxes = torch.cat((ny0, nx0, ny0 + nh, nx0 + nw), 1)
        else:
            x1 = x1 / float(image_width - 1)
            x2 = x2 / float(image_width - 1)
            y1 = y1 / float(image_height - 1)
            y2 = y2 / float(image_height - 1)
            boxes = torch.cat((y1, x1, y2, x2), 1)

        boxes = boxes.detach().contiguous()
        box_ind = box_ind.detach()
        return CropAndResizeFunction.apply(featuremap, boxes, box_ind, self.crop_height, self.crop_width, self.extrapolation_value)




def roi_align(inputs, pool_size, image_shape):
    """
    This implemnts ROI pooling on multiple levels of the feature pyramid.
    :param inputs:
    :param pool_size:
    :param image_shape:
    :return: Pooled regions [num_boxes, height, width, channels]
    """
    for i in range(len(inputs)):
        inputs[i] = inputs[i].squeeze(0)

    # crop boxes
    boxes = inputs[0]
    # feature pyramid. Each is [batch, height, width, channels]
    feature_maps = inputs[1:]

    # ROI assignment to level of pyramid
    y1,x1,y2,x2 = boxes.chunk(4,dim=1)
    h = y2-y1
    w = x2-x1

    # this equation comes from the feature pyramid networks paper
    image_area = Variable(torch.FloatTensor([float(image_shape[0] * image_shape[1])]), requires_grad=False)
    if boxes.is_cuda:
        image_area = image_area.cuda()
    roi_level = 4 + log2(torch.sqrt(h*w)/(224.0/torch.sqrt(image_area)))
    roi_level = roi_level.round().int()
    roi_level = roi_level.clamp(2,5)

    # now loop through each level and apply pooling
    pooled = []
    box_to_lev = []
    for i, level in enumerate(range(2,6)):
        ix = roi_level==level
        if not ix.any():
            continue
        ix = torch.nonzero(ix)[:,0]
        level_boxes = boxes[ix.data, :]

        # keep track of which box is in which level
        box_to_lev.append(ix.data)

        # stop gradient prop
        level_boxes = level_boxes.detach()

        # From Mask R-CNN paper: "We sample four regular locations, so
        # that we can evaluate either max or average pooling. In fact,
        # interpolating only a single value at each bin center (without
        # pooling) is nearly as effective."

        ind = Variable(torch.zeros(level_boxes.size()[0]), requires_grad=False)
        ind = ind.int()
        if level_boxes.is_cuda:
            ind = ind.cuda()
        # to fit inside of Crop and Resize
        feature_maps[i] = feature_maps[i].unsqueeze(0)
        pooled_feats


