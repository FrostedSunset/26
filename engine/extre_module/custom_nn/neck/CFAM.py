import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops

import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.core import register
from engine.extre_module.ultralytics_nn.conv import Conv, autopad
from engine.extre_module.ultralytics_nn.block import C2f

__all__ = ['CFAM']


class ADown(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)
    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1,x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class FocusFeature(nn.Module):
    def __init__(self, inc, kernel_sizes=(5, 7, 9, 11), e=0.5) -> None:
        super().__init__()
        hidc = int(inc[1] * e)

        self.conv1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv(inc[0], hidc, 1)
        )
        self.conv2 = Conv(inc[1], hidc, 1) if e != 1 else nn.Identity()
        self.conv3 = ADown(inc[2], hidc)

        self.dw_conv = nn.ModuleList(nn.Conv2d(hidc * 3, hidc * 3, kernel_size=k, padding=autopad(k), groups=hidc * 3) for k in kernel_sizes)
        self.pw_conv = Conv(hidc * 3, hidc * 3)
        self.conv_1x1 = Conv(hidc * 3, int(hidc / e))

    def forward(self, x):
        x1, x2, x3 = x
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)

        x = torch.cat([x1, x2, x3], dim=1)
        feature = torch.sum(torch.stack([x] + [layer(x) for layer in self.dw_conv], dim=0), dim=0)
        feature = self.pw_conv(feature)

        x = x + feature
        return self.conv_1x1(x)


@register(force=True)
class CFAM(nn.Module):
    def __init__(
            self,
            in_channels=[512, 1024, 2048],
            feat_strides=[8, 16, 32],
            hidden_dim=256,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.0,
            enc_act='gelu',
            use_encoder_idx=[2],
            num_encoder_layers=1,
            pe_temperature=10000,
            CFAM_ks=[3, 5, 7, 9], 
            depth_mult=1.0,
            out_strides=[8, 16, 32],
            eval_spatial_size=None,
    ):
        super().__init__()
        from engine.deim.hybrid_encoder import TransformerEncoderLayer, TransformerEncoder
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = out_strides

        assert len(in_channels) == 3

        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            proj = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                ('norm', nn.BatchNorm2d(hidden_dim))
            ]))
            self.input_proj.append(proj)

        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act
        )
        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers)
            for _ in range(len(use_encoder_idx))
        ])

        self.FocusFeature_1 = FocusFeature(inc=[hidden_dim, hidden_dim, hidden_dim], kernel_sizes=CFAM_ks)

        self.p4_to_p5_down1 = Conv(hidden_dim, hidden_dim, k=3, s=2)
        self.p5_block1 = C2f(hidden_dim * 2, hidden_dim, round(3 * depth_mult), shortcut=True)

        self.p4_to_p3_up1 = nn.Upsample(scale_factor=2)
        self.p3_block1 = C2f(hidden_dim * 2, hidden_dim, round(3 * depth_mult), shortcut=True)

        self.FocusFeature_2 = FocusFeature(inc=[hidden_dim, hidden_dim, hidden_dim], kernel_sizes=CFAM_ks)

        self.p4_to_p5_down2 = Conv(hidden_dim, hidden_dim, k=3, s=2)
        self.p5_block2 = C2f(hidden_dim * 3, hidden_dim, round(3 * depth_mult), shortcut=True)
        if len(out_strides) == 3:
            self.p4_to_p3_up2 = nn.Upsample(scale_factor=2)
            self.p3_block2 = C2f(hidden_dim * 3, hidden_dim, round(3 * depth_mult), shortcut=True)

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride,
                    self.eval_spatial_size[0] // stride,
                    self.hidden_dim,
                    self.pe_temperature
                )
                setattr(self, f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]
        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
        fouce_feature1 = self.FocusFeature_1(proj_feats[::-1])

        fouce_feature1_to_p5_1 = self.p4_to_p5_down1(fouce_feature1)
        fouce_feature1_to_p5_2 = self.p5_block1(torch.cat([fouce_feature1_to_p5_1, proj_feats[2]], dim=1))
        fouce_feature1_to_p3_1 = self.p4_to_p3_up1(fouce_feature1)
        fouce_feature1_to_p3_2 = self.p3_block1(torch.cat([fouce_feature1_to_p3_1, proj_feats[0]], dim=1))
        fouce_feature2 = self.FocusFeature_2([fouce_feature1_to_p5_2, fouce_feature1, fouce_feature1_to_p3_2])

        fouce_feature2_to_p5 = self.p4_to_p5_down2(fouce_feature2)
        fouce_feature2_to_p5 = self.p5_block2(torch.cat([fouce_feature2_to_p5, fouce_feature1_to_p5_1, fouce_feature1_to_p5_2], dim=1))
        if len(self.out_strides) == 3:
            fouce_feature2_to_p3 = self.p4_to_p3_up2(fouce_feature2)
            fouce_feature2_to_p3 = self.p3_block2(torch.cat([fouce_feature2_to_p3, fouce_feature1_to_p3_1, fouce_feature1_to_p3_2], dim=1))
            return [fouce_feature2_to_p3, fouce_feature2, fouce_feature2_to_p5]
        else:
            return [fouce_feature2, fouce_feature2_to_p5]


if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    bs, image_height, image_width = 1, 640, 640
    hidden_dim = 128
    params = {
        'in_channels': [32, 64, 128],
        'feat_strides': [8, 16, 32],
        'hidden_dim': hidden_dim,
        'use_encoder_idx': [2],
        'CFAM_ks': [3, 5, 7, 9],
        'depth_mult': 1.0,
        'out_strides': [16, 32],
        'eval_spatial_size': [image_height, image_width]
    }

    feats = [
        torch.randn((bs, hidden_dim, image_height // 16, image_width // 16)).to(device),
        torch.randn((bs, hidden_dim, image_height // 32, image_width // 32)).to(device)
    ]
    module = CFAM(**params).to(device)
    outputs = module(feats)

    input_feats_info = ', '.join([str(i.size()) for i in feats])
    print(GREEN + f'input feature:[{input_feats_info}]' + RESET)
    output_feats_info = ', '.join([str(i.size()) for i in outputs])
    print(GREEN + f'output feature:[{output_feats_info}]' + RESET)

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                    args=[feats],
                                    output_as_string=True,
                                    output_precision=4,
                                    print_detailed=True)
    print(RESET)
