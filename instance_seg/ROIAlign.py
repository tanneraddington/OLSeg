import math
import numpy as np
from torch.autograd import Variable
import torch
from pytorch_utils import log2

##from roialign.roi_align.crop_and_resize import CropAndResizeFunction

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


