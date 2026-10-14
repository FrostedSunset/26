# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
from typing import List
from torch import Tensor
from engine.core import register

def build_norm_layer(norm_type, dim):
    if norm_type == nn.BatchNorm2d or norm_type is nn.BatchNorm2d:
        return ("bn", nn.BatchNorm2d(dim))
    else:
        raise NotImplementedError


@register(force=True)
class QuantileHead(nn.Module):
    """
    Quantile Detection Head for conformal prediction
    Output: cls_logits, box_pred, box_quantiles (lower, median, upper)
    """
    def __init__(
        self,
        in_channels: List[int],
        num_classes: int,
        num_quantiles: int = 3,  # τ=[0.05,0.5,0.95]
        hidden_dim: int = 256,
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_quantiles = num_quantiles
        self.hidden_dim = hidden_dim

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.quantile_convs = nn.ModuleList()

        for c in in_channels:
            cls_branch = nn.Sequential(
                nn.Conv2d(c, hidden_dim, 3, padding=1, bias=False),
                build_norm_layer(norm_layer, hidden_dim)[1],
                act_layer(),
                nn.Conv2d(hidden_dim, num_classes, 3, padding=1)
            )
            reg_branch = nn.Sequential(
                nn.Conv2d(c, hidden_dim, 3, padding=1, bias=False),
                build_norm_layer(norm_layer, hidden_dim)[1],
                act_layer(),
                nn.Conv2d(hidden_dim, 4, 3, padding=1)
            )
            q_branch = nn.Sequential(
                nn.Conv2d(c, hidden_dim, 3, padding=1, bias=False),
                build_norm_layer(norm_layer, hidden_dim)[1],
                act_layer(),
                nn.Conv2d(hidden_dim, 4 * num_quantiles, 3, padding=1)
            )
            self.cls_convs.append(cls_branch)
            self.reg_convs.append(reg_branch)
            self.quantile_convs.append(q_branch)

    def forward(self, feats: List[Tensor]):
        cls_outputs = []
        reg_outputs = []
        quantile_outputs = []
        for feat, cls_conv, reg_conv, q_conv in zip(feats, self.cls_convs, self.reg_convs, self.quantile_convs):
            cls_out = cls_conv(feat)
            reg_out = reg_conv(feat)
            q_out = q_conv(feat)
            cls_outputs.append(cls_out)
            reg_outputs.append(reg_out)
            quantile_outputs.append(q_out)
        return cls_outputs, reg_outputs, quantile_outputs
