import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops

import copy
from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.core import register
from engine.extre_module.ultralytics_nn.conv import Conv, autopad
from engine.extre_module.ultralytics_nn.block import C2f

__all__ = ['ConformalCPHead']


def pinball_loss(y, q_pred, tau):
    diff = y - q_pred
    loss = torch.maximum(tau * diff, (tau - 1) * diff)
    return loss.mean()


class QuantileBranch(nn.Module):
    """Lightweight two-layer MLP quantile branch for single coordinate, same as paper"""
    def __init__(self, feat_dim, hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2)
        )

    def forward(self, x):
        # return [q_low, q_high]
        return self.mlp(x)


@register(force=True)
class ConformalCPHead(nn.Module):
    """
    Joint Conformal Prediction Head for DEIM, align with Algorithm in your paper
    Input: multi-scale feature list (output from FDPN)
    Workflow:
        1. Training: compute pinball loss for 4 bbox coordinates
        2. Calibration: compute conformity scores E_reg, E_cls
        3. Inference: output calibrated bbox interval, classification prediction set, uncertainty U(X)
    """
    def __init__(
            self,
            hidden_dim=256,
            feat_idx=1,       # which feature from FDPN output list to use, default P4
            num_classes=5,
            alpha=0.1,
            mlp_hidden=128
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.feat_idx = feat_idx
        self.num_classes = num_classes
        self.alpha = alpha
        self.tau_low = alpha / 2.0
        self.tau_high = 1.0 - alpha / 2.0

        # Quantile heads for x1, y1, x2, y2
        self.quantile_heads = nn.ModuleDict({
            "x1": QuantileBranch(hidden_dim, mlp_hidden),
            "y1": QuantileBranch(hidden_dim, mlp_hidden),
            "x2": QuantileBranch(hidden_dim, mlp_hidden),
            "y2": QuantileBranch(hidden_dim, mlp_hidden)
        })

        # conformal thresholds as buffer
        self.register_buffer("Q_reg_x1", torch.tensor(0.0))
        self.register_buffer("Q_reg_y1", torch.tensor(0.0))
        self.register_buffer("Q_reg_x2", torch.tensor(0.0))
        self.register_buffer("Q_reg_y2", torch.tensor(0.0))
        self.register_buffer("Q_cls", torch.tensor(0.0))
        self.calibrated = False

    def forward(self, feats, bbox_gt=None, cls_gt=None, is_calibrate=False):
        """
        Args:
            feats: list of feature maps, output from FDPN [B, C, H, W]
            bbox_gt: [B, 4] x1y1x2y2, None for pure inference
            cls_gt: [B,], None for pure inference
            is_calibrate: True -> calibration mode, only compute conformity score
        Returns:
            dict:
                train: loss_pin
                calibrate: E_reg dict, E_cls tensor
                inference: bbox_interval, cls_pred_set, U_score
        """
        # select target feature map and flatten spatial dimension
        feat_map = feats[self.feat_idx]
        B, C, H, W = feat_map.shape
        # global average pool to get feature vector per sample, match your MLP input
        feat_vec = F.adaptive_avg_pool2d(feat_map, (1, 1)).view(B, C)

        # forward all quantile branches
        q_out_dict = {}
        for coord in ["x1", "y1", "x2", "y2"]:
            q_out_dict[coord] = self.quantile_heads[coord](feat_vec)  # [B, 2]

        # ===================== TRAIN MODE =====================
        if self.training and (bbox_gt is not None and cls_gt is not None) and (not is_calibrate):
            loss_pin = 0.0
            gt_split = {
                "x1": bbox_gt[:, 0:1],
                "y1": bbox_gt[:, 1:2],
                "x2": bbox_gt[:, 2:3],
                "y2": bbox_gt[:, 3:4]
            }
            for coord in ["x1", "y1", "x2", "y2"]:
                q_low, q_high = q_out_dict[coord][:, 0:1], q_out_dict[coord][:, 1:2]
                y = gt_split[coord]
                l_low = pinball_loss(y, q_low, self.tau_low)
                l_high = pinball_loss(y, q_high, self.tau_high)
                loss_pin = loss_pin + l_low + l_high
            return {"loss_pin": loss_pin}

        # ===================== CALIBRATION MODE =====================
        if is_calibrate and (bbox_gt is not None and cls_gt is not None):
            self.eval()
            with torch.no_grad():
                E_reg_dict = {}
                gt_split = {
                    "x1": bbox_gt[:, 0],
                    "y1": bbox_gt[:, 1],
                    "x2": bbox_gt[:, 2],
                    "y2": bbox_gt[:, 3]
                }
                for coord in ["x1", "y1", "x2", "y2"]:
                    q_low, q_high = q_out_dict[coord][:, 0], q_out_dict[coord][:, 1]
                    y = gt_split[coord]
                    E_reg = torch.maximum(q_low - y, y - q_high)
                    E_reg_dict[coord] = E_reg

                # classification conformity score: E_cls = 1 - S(y_gt)
                # note: cls_score should come from DEIM cls head, pass externally in your full pipeline
                # here we only reserve interface; you combine with DEIM cls_score outside this head
                raise RuntimeError("Calibration needs classification score from DEIM head, collect outside!")
            # return {"E_reg": E_reg_dict, "E_cls": E_cls}

        # ===================== INFERENCE MODE =====================
        self.eval()
        assert self.calibrated, "Run calibration and assign thresholds first!"
        with torch.no_grad():
            interval_dict = {}
            Q_reg_map = {
                "x1": self.Q_reg_x1,
                "y1": self.Q_reg_y1,
                "x2": self.Q_reg_x2,
                "y2": self.Q_reg_y2
            }
            for coord in ["x1", "y1", "x2", "y2"]:
                q_low, q_high = q_out_dict[coord][:, 0], q_out_dict[coord][:, 1]
                qr = Q_reg_map[coord]
                int_left = q_low - qr
                int_right = q_high + qr
                interval_dict[coord] = torch.stack([int_left, int_right], dim=-1)

            # U(X) per coordinate
            U_dict = {}
            for coord in ["x1", "y1", "x2", "y2"]:
                q_low, q_high = q_out_dict[coord][:, 0], q_out_dict[coord][:, 1]
                E_reg_x = torch.maximum(q_low - (q_low + q_high) / 2, (q_low + q_high) / 2 - q_high)
                qr = Q_reg_map[coord]
                Uc = torch.minimum(torch.minimum(qr, self.Q_cls), E_reg_x)
                U_dict[coord] = Uc

        return {
            "bbox_interval": interval_dict,
            "U_score": U_dict,
            "feat_vec": feat_vec
        }

    def assign_calibrated_thresholds(self, E_reg_all, E_cls_all):
        """
        Collect all conformity scores on calibration set, compute quantile and assign to buffer
        Args:
            E_reg_all: dict{"x1":np.array, "y1":np.array, "x2":np.array, "y2":np.array}
            E_cls_all: np.array
        """
        N_cal = len(E_cls_all)
        k = int(np.ceil((1 - self.alpha) * (N_cal + 1)))

        self.Q_reg_x1.copy_(torch.tensor(np.sort(E_reg_all["x1"])[k - 1]))
        self.Q_reg_y1.copy_(torch.tensor(np.sort(E_reg_all["y1"])[k - 1]))
        self.Q_reg_x2.copy_(torch.tensor(np.sort(E_reg_all["x2"])[k - 1]))
        self.Q_reg_y2.copy_(torch.tensor(np.sort(E_reg_all["y2"])[k - 1]))
        self.Q_cls.copy_(torch.tensor(np.sort(E_cls_all)[k - 1]))
        self.calibrated = True


if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    bs, image_height, image_width = 1, 640, 640
    hidden_dim = 128
    # simulate FDPN output feats
    params_fdpn = {
        'in_channels': [32, 64, 128],
        'feat_strides': [8, 16, 32],
        'hidden_dim': hidden_dim,
        'use_encoder_idx': [2],
        'fdpn_ks': [3, 5, 7, 9],
        'depth_mult': 1.0,
        'out_strides': [16, 32],
        'eval_spatial_size': [image_height, image_width]
    }
    feats = [
        torch.randn((bs, hidden_dim, image_height // 16, image_width // 16)).to(device),
        torch.randn((bs, hidden_dim, image_height // 32, image_width // 32)).to(device)
    ]
    cp_params = {
        "hidden_dim": hidden_dim,
        "feat_idx": 0,
        "num_classes": 5,
        "alpha": 0.1
    }
    module = ConformalCPHead(**cp_params).to(device)
    dummy_bbox = torch.randn(bs, 4).to(device)
    dummy_cls = torch.randint(0, 5, (bs,)).to(device)

    out_train = module(feats, dummy_bbox, dummy_cls, is_calibrate=False)
    print(GREEN + f"train loss_pin: {out_train['loss_pin'].item():.4f}" + RESET)

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     args=[feats, dummy_bbox, dummy_cls, False],
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)
