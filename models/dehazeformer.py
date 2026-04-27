"""
DehazeFormer - Swin Transformer 기반 안개 제거 모델
Reference: Song et al. (2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math


# ────────────────────────────────────────────
# 기본 빌딩 블록
# ────────────────────────────────────────────

class RescaleNorm(nn.Module):
    """DehazeFormer 핵심: LayerNorm + 학습 가능한 스케일 파라미터"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.norm(x) * self.scale


class SKFusion(nn.Module):
    """Selective Kernel Fusion: 두 feature를 동적으로 융합"""
    def __init__(self, dim, height=2, reduction=8):
        super().__init__()
        self.height = height
        d = max(dim // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feats):
        B, C, H, W = feats[0].shape
        feats_sum = sum(feats)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.reshape(B, self.height, C, 1, 1))
        return sum(f * a for f, a in zip(feats, attn.unbind(dim=1)))


class WindowAttention(nn.Module):
    """Window-based Multi-head Self Attention (W-MSA)"""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = to_2tuple(window_size)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads)
        )
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        attn = attn + relative_position_bias.permute(2, 0, 1).unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block (W-MSA / SW-MSA)"""
    def __init__(self, dim, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = RescaleNorm(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads,
                                    qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = RescaleNorm(dim)

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop)
        )

    def forward(self, x, attn_mask=None):
        B, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        x_windows = window_partition(x, self.window_size).view(-1, self.window_size ** 2, C)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        x = window_reverse(attn_windows.view(-1, self.window_size, self.window_size, C), self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ────────────────────────────────────────────
# Patch Embedding / Unembedding
# ────────────────────────────────────────────

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(x)


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, embed_dim=96, out_chans=3):
        super().__init__()
        self.proj = nn.ConvTranspose2d(embed_dim, out_chans, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, hw):
        H, W = hw
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        return self.proj(x)


# ────────────────────────────────────────────
# Encoder / Decoder 스테이지
# ────────────────────────────────────────────

class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=4.,
                 qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path
            ) for i in range(depth)
        ])

    def forward(self, x):
        B, H, W, C = x.shape
        # attn_mask 계산 생략 (간소화)
        for blk in self.blocks:
            x = blk(x)
        return x


# ────────────────────────────────────────────
# DehazeFormer 메인 모델
# ────────────────────────────────────────────

class DehazeFormer(nn.Module):
    """
    DehazeFormer: Swin Transformer 기반 단일 이미지 안개 제거
    sizes: T(tiny), S(small), B(base), L(large)
    """
    CONFIGS = {
        'T': dict(embed_dim=48,  depths=[2,2,2,2], num_heads=[3,6,12,24]),
        'S': dict(embed_dim=48,  depths=[2,2,6,2], num_heads=[3,6,12,24]),
        'B': dict(embed_dim=64,  depths=[2,2,6,2], num_heads=[4,8,16,32]),
        'L': dict(embed_dim=96,  depths=[2,2,18,2], num_heads=[6,12,24,48]),
    }

    def __init__(self, size='B', in_chans=3, window_size=8, mlp_ratio=4.,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1):
        super().__init__()
        cfg = self.CONFIGS[size]
        embed_dim = cfg['embed_dim']
        depths = cfg['depths']
        num_heads = cfg['num_heads']

        self.patch_embed = PatchEmbed(patch_size=1, in_chans=in_chans, embed_dim=embed_dim)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Encoder
        self.encoder1 = BasicLayer(embed_dim,     depths[0], num_heads[0], window_size,
                                   mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate,
                                   drop_path=dpr[:depths[0]])
        self.down1 = nn.Conv2d(embed_dim, embed_dim*2, 2, 2)

        self.encoder2 = BasicLayer(embed_dim*2,   depths[1], num_heads[1], window_size,
                                   mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate,
                                   drop_path=dpr[depths[0]:sum(depths[:2])])
        self.down2 = nn.Conv2d(embed_dim*2, embed_dim*4, 2, 2)

        # Bottleneck
        self.bottleneck = BasicLayer(embed_dim*4, depths[2], num_heads[2], window_size,
                                     mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate,
                                     drop_path=dpr[sum(depths[:2]):sum(depths[:3])])

        # Decoder
        self.up2 = nn.ConvTranspose2d(embed_dim*4, embed_dim*2, 2, 2)
        self.fusion2 = SKFusion(embed_dim*2)
        self.decoder2 = BasicLayer(embed_dim*2,   depths[1], num_heads[1], window_size,
                                   mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate,
                                   drop_path=dpr[sum(depths[:1]):sum(depths[:2])])

        self.up1 = nn.ConvTranspose2d(embed_dim*2, embed_dim, 2, 2)
        self.fusion1 = SKFusion(embed_dim)
        self.decoder1 = BasicLayer(embed_dim,     depths[0], num_heads[0], window_size,
                                   mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate,
                                   drop_path=dpr[:depths[0]])

        self.patch_unembed = PatchUnEmbed(patch_size=1, embed_dim=embed_dim, out_chans=in_chans)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        B, C, H, W = x.shape

        # Patch embed → (B, H, W, C)
        feat = self.patch_embed(x).view(B, H, W, -1)

        # Encoder
        e1 = self.encoder1(feat)                                        # B,H,W,C
        e1_c = e1.permute(0,3,1,2)                                      # B,C,H,W
        e2 = self.encoder2(self.down1(e1_c).permute(0,2,3,1))          # B,H/2,W/2,2C
        e2_c = e2.permute(0,3,1,2)

        # Bottleneck
        b = self.bottleneck(self.down2(e2_c).permute(0,2,3,1))
        b_c = b.permute(0,3,1,2)

        # Decoder
        d2 = self.decoder2(self.fusion2([self.up2(b_c), e2_c]).permute(0,2,3,1))
        d2_c = d2.permute(0,3,1,2)
        d1 = self.decoder1(self.fusion1([self.up1(d2_c), e1_c]).permute(0,2,3,1))

        out = self.patch_unembed(d1.view(B, H*W, -1), (H, W))
        return out + x   # residual


def build_model(size='B', **kwargs):
    return DehazeFormer(size=size, **kwargs)
