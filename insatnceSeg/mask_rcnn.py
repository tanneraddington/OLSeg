"""
Mask R-CNN
The main Mask R-CNN model implemenetation.
Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)

Rewritten by Tanner Watts
"""
import datetime
import math
import os
import random
import re

import numpy as np
import torch
import torch.nn as nn
import torchvision.transform.functional as TF

from roialign.roi_align.crop_and_resize import CropAndResizeFunction
import utils


class Mask_rcnn():
    """
    This is the mask rcnn model.
    """
