"""
DehazeFormer + AFDA (Adaptive Fog Density Attention)
근거: GC(Deformable Attn) 실험에서 안개 비균일성 대응이 기대보다 부족했음
개선: Attention score에 안개 밀도 추정 bias를 추가하여
      안개가 짙은 픽셀(고밝기, 저채도)끼리 더 강하게 연결되도록 유도
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math


class RescaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.norm(x) * self.scale


class SKFusion(nn.Module):
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


# ── AFDA 핵심: 안개 밀도 추정기 ──────────────────────────────
class FogDensityEstimator(nn.Module):
    """
    입력 이미지에서 안개 밀도 맵을 추정.
    밝고 채도가 낮은 영역 = 안개 짙음 → 높은 값 출력
    window_size x window_size 패치 단위로 fog_bias 생성
    """
    def __init__(self, window_size=8):
        super().__init__()
        self.window_size = window_size
        # 안개 밀도 추정 경량 CNN
        self.estimator = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()  # 0~1 범위로 정규화
        )
        self.alpha = nn.Parameter(torch.tensor(0.1))  # 학습 가능한 bias 강도

    def forward(self, x_img, H, W, num_heads):
        """
        x_img: (B, 3, H_orig, W_orig) 원본 입력 이미지
        H, W: feature map 크기
        반환: (B*num_windows, num_heads, N, N) attention bias
        """
        B = x_img.shape[0]
        ws = self.window_size
        N = ws * ws

        # 안개 밀도 맵 추정 (B, 1, H_orig, W_orig)
        fog_map = self.estimator(x_img)

        # feature map 크기로 리사이즈
        fog_map = F.interpolate(fog_map, size=(H, W), mode='bilinear', align_corners=False)

        # 윈도우 단위로 분할 → (B * nW, N)
        fog_map = fog_map.squeeze(1)  # (B, H, W)
        nH = H // ws
        nW_win = W // ws
        fog_map = fog_map.view(B, nH, ws, nW_win, ws)
        fog_map = fog_map.permute(0, 1, 3, 2, 4).contiguous()
        fog_map = fog_map.view(B * nH * nW_win, N)  # (B*nW, N)

        # 외적으로 (N, N) bias 생성: 두 픽셀 모두 안개 짙으면 bias 높음
        fog_i = fog_map.unsqueeze(2)  # (B*nW, N, 1)
        fog_j = fog_map.unsqueeze(1)  # (B*nW, 1, N)
        bias = fog_i * fog_j          # (B*nW, N, N)

        # num_heads 차원 추가 + alpha 스케일링
        bias = bias.unsqueeze(1).expand(-1, num_heads, -1, -1)  # (B*nW, nH, N, N)
        return self.alpha * bias


class WindowAttentionAFDA(nn.Module):
    """
    AFDA가 적용된 Window Attention
    기존 attention score에 fog density bias 추가
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = to_2tuple(window_size)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2*self.window_size[0]-1)*(2*self.window_size[1]-1), num_heads)
        )
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:,:,None] - coords_flatten[:,None,:]
        relative_coords = relative_coords.permute(1,2,0).contiguous()
        relative_coords[:,:,0] += self.window_size[0] - 1
        relative_coords[:,:,1] += self.window_size[1] - 1
        relative_coords[:,:,0] *= 2*self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None, fog_bias=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C//self.num_heads).permute(2,0,3,1,4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2,-1)) * self.scale

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size[0]*self.window_size[1], self.window_size[0]*self.window_size[1], -1)
        attn = attn + relative_position_bias.permute(2,0,1).unsqueeze(0)

        # ── AFDA 핵심: fog bias 추가 ──
        if fog_bias is not None:
            attn = attn + fog_bias

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_//nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1,2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H//window_size, window_size, W//window_size, window_size, C)
    return x.permute(0,1,3,2,4,5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H//window_size, W//window_size, window_size, window_size, -1)
    return x.permute(0,1,3,2,4,5).contiguous().view(B, H, W, -1)


class SwinTransformerBlockAFDA(nn.Module):
    """AFDA가 적용된 Swin Transformer Block"""
    def __init__(self, dim, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = RescaleNorm(dim)
        self.attn = WindowAttentionAFDA(dim, window_size=window_size, num_heads=num_heads,
                                        qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = RescaleNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden), act_layer(), nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim), nn.Dropout(drop)
        )

    def forward(self, x, fog_bias=None):
        B, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1,2))

        x_windows = window_partition(x, self.window_size).view(-1, self.window_size**2, C)
        attn_windows = self.attn(x_windows, fog_bias=fog_bias)
        x = window_reverse(attn_windows.view(-1, self.window_size, self.window_size, C), self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1,2))

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BasicLayerAFDA(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=4.,
                 qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlockAFDA(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i%2==0) else window_size//2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path
            ) for i in range(depth)
        ])

    def forward(self, x, fog_bias=None):
        for blk in self.blocks:
            x = blk(x, fog_bias=fog_bias)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        return self.norm(self.proj(x).flatten(2).transpose(1,2))


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, embed_dim=96, out_chans=3):
        super().__init__()
        self.proj = nn.ConvTranspose2d(embed_dim, out_chans, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, hw):
        H, W = hw
        B, N, C = x.shape
        return self.proj(x.transpose(1,2).view(B, C, H, W))


class DehazeFormerAFDA(nn.Module):
    """
    DehazeFormer + AFDA (Adaptive Fog Density Attention)
    변경 사항: WindowAttention → WindowAttentionAFDA
               FogDensityEstimator 추가 (파라미터 ~0.3K 추가)
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

        self.window_size = window_size
        self.num_heads_enc1 = num_heads[0]

        self.patch_embed = PatchEmbed(patch_size=1, in_chans=in_chans, embed_dim=embed_dim)
        self.fog_estimator = FogDensityEstimator(window_size=window_size)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.encoder1 = BasicLayerAFDA(embed_dim, depths[0], num_heads[0], window_size,
                                       mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate,
                                       drop_path=dpr[:depths[0]])
        self.down1 = nn.Conv2d(embed_dim, embed_dim*2, 2, 2)

        self.encoder2 = BasicLayerAFDA(embed_dim*2, depths[1], num_heads[1], window_size,
                                       mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate,
                                       drop_path=dpr[depths[0]:sum(depths[:2])])
        self.down2 = nn.Conv2d(embed_dim*2, embed_dim*4, 2, 2)

        self.bottleneck = BasicLayerAFDA(embed_dim*4, depths[2], num_heads[2], window_size,
                                         mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate,
                                         drop_path=dpr[sum(depths[:2]):sum(depths[:3])])

        self.up2 = nn.ConvTranspose2d(embed_dim*4, embed_dim*2, 2, 2)
        self.fusion2 = SKFusion(embed_dim*2)
        self.decoder2 = BasicLayerAFDA(embed_dim*2, depths[1], num_heads[1], window_size,
                                       mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate,
                                       drop_path=dpr[sum(depths[:1]):sum(depths[:2])])

        self.up1 = nn.ConvTranspose2d(embed_dim*2, embed_dim, 2, 2)
        self.fusion1 = SKFusion(embed_dim)
        self.decoder1 = BasicLayerAFDA(embed_dim, depths[0], num_heads[0], window_size,
                                       mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate,
                                       drop_path=dpr[:depths[0]])

        self.patch_unembed = PatchUnEmbed(patch_size=1, embed_dim=embed_dim, out_chans=in_chans)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        B, C, H, W = x.shape

        # 안개 밀도 bias 계산 (encoder1 레벨에서만 적용)
        fog_bias = self.fog_estimator(x, H, W, self.num_heads_enc1)

        feat = self.patch_embed(x).view(B, H, W, -1)

        # Encoder (encoder1에 fog_bias 적용)
        e1 = self.encoder1(feat, fog_bias=fog_bias)
        e1_c = e1.permute(0,3,1,2)
        e2 = self.encoder2(self.down1(e1_c).permute(0,2,3,1))
        e2_c = e2.permute(0,3,1,2)

        # Bottleneck
        b = self.bottleneck(self.down2(e2_c).permute(0,2,3,1))
        b_c = b.permute(0,3,1,2)

        # Decoder
        d2 = self.decoder2(self.fusion2([self.up2(b_c), e2_c]).permute(0,2,3,1))
        d2_c = d2.permute(0,3,1,2)
        d1 = self.decoder1(self.fusion1([self.up1(d2_c), e1_c]).permute(0,2,3,1))

        out = self.patch_unembed(d1.view(B, H*W, -1), (H, W))
        return out + x


def build_model_afda(size='B', **kwargs):
    return DehazeFormerAFDA(size=size, **kwargs)
