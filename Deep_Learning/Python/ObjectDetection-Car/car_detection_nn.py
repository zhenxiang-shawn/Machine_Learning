"""

Author: Zhenxiang Jin (zhenxiang.shawn@zohomail.com)
"""
import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.utils.data import Dataset


class CarDetectionDataSet(Dataset):

    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir