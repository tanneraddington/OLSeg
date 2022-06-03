'''
The Goal of this script is to create functions that can generate the proposal
boxes, and region of interest.
'''
import torch
import torch.nn as nn

def apply_box_deltas(boxes, deltas):
    '''
    Applies values to given boxes
    :param boxes: [N,4]
    :param deltas: [N,4] dy dx logdh logdw
    :return:
    '''
    # center the x and y and convert
    height = boxes[:,2] - boxes[:,0]
    width  = boxes[:,3] - boxes[:,1]
    y_center = boxes[:,0] * height * 0.5
    x_center = boxes[:, 1] * width * 0.5
    # Apply deltas
    y_center += deltas[:, 0] * height
    x_center += deltas[:, 1] * width
    height *= torch.exp(deltas[:, 2])
    width *= torch.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = y_center - 0.5 * height
    x1 = x_center - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = torch.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    return result

def clip_boxes(boxes, window):
    '''
    Clap the boxes to the window
    :param boxes:
    :param window:
    :return:
    '''
    boxes = torch.stack([boxes[:, 0].clamp(float(window[0]), float(window[2])),
         boxes[:, 1].clamp(float(window[1]), float(window[3])),
         boxes[:, 2].clamp(float(window[0]), float(window[2])),
         boxes[:, 3].clamp(float(window[1]), float(window[3]))], 1)
    return boxes

def call_proposal_layer(inputs, proposal_count, nms_threshold, anchors, config=None):
    '''
    The goal of this method is to introduce the proposal layer. This layer
    applies bounding box refinement to anchors. more edits here
    '''
