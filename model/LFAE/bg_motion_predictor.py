"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""

from torch import nn
import torch
from model.LFAE.util import Encoder


class BGMotionPredictor(nn.Module):
    """
    Module for background estimation, return single transformation, parametrized as 3x3 matrix.
    """

    def __init__(self, block_expansion, num_channels, max_features, num_blocks, bg_type='zero'):
        super(BGMotionPredictor, self).__init__()
        assert bg_type in ['zero', 'shift', 'affine', 'perspective']

        self.bg_type = bg_type
        if self.bg_type != 'zero':
            self.encoder = Encoder(block_expansion, in_features=num_channels * 2, max_features=max_features,
                                   num_blocks=num_blocks)
            in_features = min(max_features, block_expansion * (2 ** num_blocks))
            if self.bg_type == 'perspective':
                self.fc = nn.Linear(in_features, 8)
                self.fc.weight.data.zero_()
                self.fc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0, 0, 0], dtype=torch.float))
            elif self.bg_type == 'affine':
                """
                仿射变化：线性变换和平移变换的叠加，包括缩放、平移、旋转、反射、错切，变换后直线还是直线，中点还是中点
                
                
                """
                self.fc = nn.Linear(in_features, 6)
                self.fc.weight.data.zero_()
                self.fc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
            elif self.bg_type == 'shift':
                self.fc = nn.Linear(in_features, 2)
                self.fc.weight.data.zero_()
                self.fc.bias.data.copy_(torch.tensor([0, 0], dtype=torch.float))

    def forward(self, source_image, driving_image):
        bs = source_image.shape[0]#获取批次大小
        out = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).type(source_image.type())#单位矩阵复制成批次大小

        if self.bg_type != 'zero':
            prediction = self.encoder(torch.cat([source_image, driving_image], dim=1))
            #最后一个特征图进行均值操作（b,c,h,w)->(b,c)
            prediction = prediction[-1].mean(dim=(2, 3)).to(next(self.parameters()).device)
            prediction = self.fc(prediction)
            if self.bg_type == 'shift':
                out[:, :2, 2] = prediction
            elif self.bg_type == 'affine':
                out[:, :2, :] = prediction.view(bs, 2, 3)
            elif self.bg_type == 'perspective':
                out[:, :2, :] = prediction[:, :6].view(bs, 2, 3)
                out[:, 2, :2] = prediction[:, 6:].view(bs, 2)

        return out
