# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch.nn.functional as F
from engine.core import register


@register(force=True)
class PinballLoss(nn.Module):
    """
    Pinball Loss (Quantile Loss)
    tau: list of quantiles, e.g. [0.05, 0.5, 0.95]
    """
    def __init__(self, tau=[0.05, 0.5, 0.95], reduction="mean"):
        super().__init__()
        self.tau = torch.tensor(tau, dtype=torch.float32)
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        pred: (B, N, 4, Q)
        target: (B, N, 4)
        """
        B, N, C = target.shape
        Q = len(self.tau)
        target = target.unsqueeze(-1).expand(-1, -1, -1, Q)
        diff = target - pred
        tau = self.tau.to(pred.device)[None, None, None, :]
        loss = torch.maximum(tau * diff, (tau - 1.0) * diff)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
