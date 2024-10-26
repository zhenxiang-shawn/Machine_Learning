"""

Author: Zhenxiang Jin (zhenxiang.shawn@zohomail.com)
"""
from typing import Dict, List, OrderedDict

import torch
import torch.nn as nn

import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.models.detection.ssd import _xavier_init


class CarDetectionDataSet(Dataset):

    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir


class SSDFeatureExtractorVGG(nn.Module):
    def __init__(self, backbone: nn.Module, highres: bool):
        super().__init__()

        # 得到maxpool3和maxpool4的位置，这里backbone是vgg16模型
        _, _, maxpool3_pos, maxpool4_pos, _ = (i for i, layer in
                                               enumerate(backbone) if
                                               isinstance(layer, nn.MaxPool2d))

        # maxpool3开启ceil_mode，这样得到的特征图大小是38x38，而不是37x37
        backbone[maxpool3_pos].ceil_mode = True

        # Conv4_3的L2 regularization + rescaling
        self.scale_weight = nn.Parameter(torch.ones(512) * 20)

        # Conv4_3之前的模块，用来提取第一个特征图
        self.features = nn.Sequential(
            *backbone[:maxpool4_pos]
        )

        # 额外增加的4个模块
        extra = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),
                # conv8_2
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
                # conv9_2
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3),  # conv10_2
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3),  # conv11_2
                nn.ReLU(inplace=True),
            )
        ])
        if highres:
            # SSD512还多了一个额外的模块
            extra.append(nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=4),  # conv12_2
                nn.ReLU(inplace=True),
            ))
        _xavier_init(extra)

        # maxpool5+Conv6(fc6)+Conv7(fc7)，这里直接随机初始化，没有转换权重
        fc = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=False),
            # add modified maxpool5
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3,
                      padding=6, dilation=6),  # FC6 with atrous
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),
            # FC7
            nn.ReLU(inplace=True)
        )
        _xavier_init(fc)
        # 添加Conv5_3，即第2个特征图
        extra.insert(0, nn.Sequential(
            *backbone[maxpool4_pos:-1],  # until conv5_3, skip maxpool5
            fc,
        ))
        self.extra = extra

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # Conv4_3
        x = self.features(x)
        rescaled = self.scale_weight.view(1, -1, 1, 1) * F.normalize(x)
        output = [rescaled]

        # 计算Conv5_3, Conv8_2，Conv9_2，Conv10_2，Conv11_2，(Conv12_2)
        for block in self.extra:
            x = block(x)
            output.append(x)

        return OrderedDict([(str(i), v) for i, v in enumerate(output)])


# 基类：实现tensor的转换，主要将4D tensor转换成最后预测格式(N, H*W*A, K)
class SSDScoringHead(nn.Module):
    def __init__(self, module_list: nn.ModuleList, num_columns: int):
        super().__init__()
        self.module_list = module_list
        self.num_columns = num_columns

    def _get_result_from_module_list(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.module_list[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.module_list)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.module_list):
            if i == idx:
                out = module(x)
        return out

    def forward(self, x: List[Tensor]) -> Tensor:
        all_results = []

        for i, features in enumerate(x):
            results = self._get_result_from_module_list(features, i)

            # Permute output from (N, A * K, H, W) to (N, HWA, K).
            N, _, H, W = results.shape
            results = results.view(N, -1, self.num_columns, H, W)
            results = results.permute(0, 3, 4, 1, 2)
            results = results.reshape(N, -1,
                                      self.num_columns)  # Size=(N, HWA, K)

            all_results.append(results)

        return torch.cat(all_results, dim=1)


# 类别预测head
class SSDClassificationHead(SSDScoringHead):
    def __init__(self, in_channels: List[int], num_anchors: List[int],
                 num_classes: int):
        cls_logits = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            cls_logits.append(
                nn.Conv2d(channels, num_classes * anchors, kernel_size=3,
                          padding=1))
        _xavier_init(cls_logits)
        super().__init__(cls_logits, num_classes)


# 位置回归head
class SSDRegressionHead(SSDScoringHead):
    def __init__(self, in_channels: List[int], num_anchors: List[int]):
        bbox_reg = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            bbox_reg.append(
                nn.Conv2d(channels, 4 * anchors, kernel_size=3, padding=1))
        _xavier_init(bbox_reg)
        super().__init__(bbox_reg, 4)


class SSDHead(nn.Module):
    def __init__(self, in_channels: List[int], num_anchors: List[int],
                 num_classes: int):
        super().__init__()
        self.classification_head = SSDClassificationHead(in_channels,
                                                         num_anchors,
                                                         num_classes)
        self.regression_head = SSDRegressionHead(in_channels, num_anchors)

    def forward(self, x: List[Tensor]) -> Dict[str, Tensor]:
        return {
            "bbox_regression": self.regression_head(x),
            "cls_logits": self.classification_head(x),
        }