"""
평가 지표: PSNR, SSIM
"""

import torch
import torch.nn.functional as F
from pytorch_msssim import ssim as ms_ssim


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(max_val ** 2 / mse).item()


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    return ms_ssim(pred, target, data_range=1.0, size_average=True).item()
