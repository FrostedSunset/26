# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch


def compute_conformal_threshold(
    lower_pred: torch.Tensor,
    upper_pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.1
):
    """
    Conformal threshold calculation for box regression
    Args:
        lower_pred: (N,4)
        upper_pred: (N,4)
        target: (N,4)
        alpha: significance level, 1-alpha = coverage
    Return:
        q: scalar threshold
    """
    # nonconformity score
    score = torch.maximum(target - upper_pred, lower_pred - target)
    score = score.max(dim=-1)[0]
    n = score.shape[0]
    q_level = torch.ceil((n + 1) * (1 - alpha)) / n
    q = torch.quantile(score, q_level, interpolation="higher")
    return q


def apply_conformal_interval(lower_pred, upper_pred, q):
    """expand interval by conformal threshold q"""
    new_lower = lower_pred - q
    new_upper = upper_pred + q
    return new_lower, new_upper


def calc_picp_mpiw(lower, upper, target):
    """
    PICP: Prediction Interval Coverage Probability
    MPIW: Mean Prediction Interval Width
    lower/upper/target: (N,4)
    """
    inside = torch.logical_and(target >= lower, target <= upper)
    picp = inside.float().mean()
    width = (upper - lower).mean()
    return picp, width
