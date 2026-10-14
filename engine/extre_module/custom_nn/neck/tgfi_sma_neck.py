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

class SMA(nn.Module):
    """Spatial Multi-scale Aggregation Block (SMA)"""
    def __init__(self, dim, dilation_rates=[1,2,4,8], norm_layer=nn.BatchNorm2d, act_layer=nn.GELU):
        super().__init__()
        self.branches = nn.ModuleList()
        for d in dilation_rates:
            self.branches.append(nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=d, dilation=d, groups=dim, bias=False),
                build_norm_layer(norm_layer, dim)[1],
                act_layer()
            ))
        self.fusion_conv = nn.Conv2d(dim * len(dilation_rates), dim, 1, bias=False)
        self.norm = build_norm_layer(norm_layer, dim)[1]

    def forward(self, x: Tensor):
        feats = [br(x) for br in self.branches]
        x_cat = torch.cat(feats, dim=1)
        out = self.fusion_conv(x_cat)
        out = self.norm(out)
        return out


class TGFI(nn.Module):
    """Temporal-Guided Feature Interaction Block (TGFI)"""
    def __init__(self, dim, norm_layer=nn.BatchNorm2d, act_layer=nn.GELU):
        super().__init__()
        self.proj_q = nn.Conv2d(dim, dim, 1, bias=False)
        self.proj_k = nn.Conv2d(dim, dim, 1, bias=False)
        self.proj_v = nn.Conv2d(dim, dim, 1, bias=False)
        self.att_conv = nn.Conv2d(dim, dim, 1, bias=False)
        self.norm = build_norm_layer(norm_layer, dim)[1]
        self.act = act_layer()

    def forward(self, feat_list: List[Tensor]) -> List[Tensor]:
        out_list = []
        for x in feat_list:
            q = self.proj_q(x)
            k = self.proj_k(x)
            v = self.proj_v(x)
            att = torch.sigmoid(self.att_conv(q * k))
            x_out = x + att * v
            x_out = self.act(self.norm(x_out))
            out_list.append(x_out)
        return out_list


@register(force=True)
class TGFISMANeck(nn.Module):
    def __init__(self,
                 in_channels: List[int],
                 fdpn_ks=[1,2,4,8],
                 norm_layer=nn.BatchNorm2d,
                 act_layer=nn.GELU):
        super().__init__()
        self.tgfi = TGFI(dim=in_channels[0], norm_layer=norm_layer, act_layer=act_layer)
        self.sma_blocks = nn.ModuleList([
            SMA(dim=c, dilation_rates=fdpn_ks, norm_layer=norm_layer, act_layer=act_layer)
            for c in in_channels
        ])

    def forward(self, feats: List[Tensor]) -> List[Tensor]:
        feats = self.tgfi(feats)
        out_feats = []
        for f, sma in zip(feats, self.sma_blocks):
            out_feats.append(sma(f))
        return out_feats
