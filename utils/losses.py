"""
Loss 함수: L1 + Perceptual + SSIM + Frequency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch_msssim import ssim


class PerceptualLoss(nn.Module):
    """VGG-16 기반 Perceptual Loss"""
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.features = nn.Sequential(*list(vgg.features)[:16]).eval()
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, pred, target):
        return F.l1_loss(self.features(pred), self.features(target))


class FrequencyLoss(nn.Module):
    """FFT 기반 주파수 도메인 Loss (LDFormer/ACM2025 아이디어)"""
    def forward(self, pred, target):
        pred_fft   = torch.fft.rfft2(pred,   norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')
        return F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))


class DehazeFormerLoss(nn.Module):
    def __init__(self, l1_w=1.0, perceptual_w=0.0, ssim_w=0.0, freq_w=0.0):
        super().__init__()
        self.l1_w          = l1_w
        self.perceptual_w  = perceptual_w
        self.ssim_w        = ssim_w
        self.freq_w        = freq_w

        if perceptual_w > 0:
            self.perceptual = PerceptualLoss()
        if freq_w > 0:
            self.freq = FrequencyLoss()

    def forward(self, pred, target):
        loss = self.l1_w * F.l1_loss(pred, target)

        if self.perceptual_w > 0:
            loss = loss + self.perceptual_w * self.perceptual(pred, target)

        if self.ssim_w > 0:
            loss = loss + self.ssim_w * (1 - ssim(pred, target, data_range=1.0))

        if self.freq_w > 0:
            loss = loss + self.freq_w * self.freq(pred, target)

        return loss
