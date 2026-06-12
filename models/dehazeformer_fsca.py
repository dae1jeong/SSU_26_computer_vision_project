"""
DehazeFormer + FSCA v2 (Frequency-Spatial Cross Attention)
v2 변경: Cross-Attention을 전체 시퀀스 대신 채널 수준(Global)으로 변경
         4096² attention → 채널 단위 Global Avg로 컨텍스트 추출 후 FiLM 변조
         → 메모리 O(HW*C) → O(C²)로 감소
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
    def forward(self, x): return self.norm(x) * self.scale

class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super().__init__()
        self.height = height; d = max(dim//reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(nn.Conv2d(dim,d,1,bias=False), nn.ReLU(), nn.Conv2d(d,dim*height,1,bias=False))
        self.softmax = nn.Softmax(dim=1)
    def forward(self, feats):
        B,C,H,W = feats[0].shape
        attn = self.softmax(self.mlp(self.avg_pool(sum(feats))).reshape(B,self.height,C,1,1))
        return sum(f*a for f,a in zip(feats, attn.unbind(dim=1)))

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim=dim; self.window_size=to_2tuple(window_size); self.num_heads=num_heads
        head_dim=dim//num_heads; self.scale=head_dim**-0.5
        self.relative_position_bias_table=nn.Parameter(torch.zeros((2*self.window_size[0]-1)*(2*self.window_size[1]-1),num_heads))
        coords_h=torch.arange(self.window_size[0]); coords_w=torch.arange(self.window_size[1])
        coords=torch.stack(torch.meshgrid(coords_h,coords_w,indexing='ij'))
        coords_flatten=torch.flatten(coords,1)
        rc=coords_flatten[:,:,None]-coords_flatten[:,None,:]
        rc=rc.permute(1,2,0).contiguous()
        rc[:,:,0]+=self.window_size[0]-1; rc[:,:,1]+=self.window_size[1]-1
        rc[:,:,0]*=2*self.window_size[1]-1
        self.register_buffer("relative_position_index",rc.sum(-1))
        self.qkv=nn.Linear(dim,dim*3,bias=qkv_bias); self.attn_drop=nn.Dropout(attn_drop)
        self.proj=nn.Linear(dim,dim); self.proj_drop=nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table,std=.02); self.softmax=nn.Softmax(dim=-1)
    def forward(self, x, mask=None):
        B_,N,C=x.shape
        qkv=self.qkv(x).reshape(B_,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4)
        q,k,v=qkv.unbind(0); attn=(q@k.transpose(-2,-1))*self.scale
        rpb=self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0]*self.window_size[1],self.window_size[0]*self.window_size[1],-1)
        attn=attn+rpb.permute(2,0,1).unsqueeze(0)
        if mask is not None:
            nW=mask.shape[0]
            attn=attn.view(B_//nW,nW,self.num_heads,N,N)+mask.unsqueeze(1).unsqueeze(0)
            attn=attn.view(-1,self.num_heads,N,N)
        return self.proj_drop(self.proj((self.attn_drop(self.softmax(attn))@v).transpose(1,2).reshape(B_,N,C)))

def window_partition(x, ws):
    B,H,W,C=x.shape
    return x.view(B,H//ws,ws,W//ws,ws,C).permute(0,1,3,2,4,5).contiguous().view(-1,ws,ws,C)
def window_reverse(windows, ws, H, W):
    B=int(windows.shape[0]/(H*W/ws/ws))
    return windows.view(B,H//ws,W//ws,ws,ws,-1).permute(0,1,3,2,4,5).contiguous().view(B,H,W,-1)

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, shift_size=0, mlp_ratio=4.,
                 qkv_bias=True, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.window_size=window_size; self.shift_size=shift_size
        self.norm1=RescaleNorm(dim); self.attn=WindowAttention(dim,window_size,num_heads,qkv_bias,attn_drop,drop)
        self.drop_path=DropPath(drop_path) if drop_path>0. else nn.Identity()
        self.norm2=RescaleNorm(dim)
        mlp_h=int(dim*mlp_ratio)
        self.mlp=nn.Sequential(nn.Linear(dim,mlp_h),act_layer(),nn.Dropout(drop),nn.Linear(mlp_h,dim),nn.Dropout(drop))
    def forward(self, x):
        B,H,W,C=x.shape; sc=x; x=self.norm1(x)
        if self.shift_size>0: x=torch.roll(x,shifts=(-self.shift_size,-self.shift_size),dims=(1,2))
        xw=window_partition(x,self.window_size).view(-1,self.window_size**2,C)
        x=window_reverse(self.attn(xw).view(-1,self.window_size,self.window_size,C),self.window_size,H,W)
        if self.shift_size>0: x=torch.roll(x,shifts=(self.shift_size,self.shift_size),dims=(1,2))
        x=sc+self.drop_path(x); return x+self.drop_path(self.mlp(self.norm2(x)))

class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=4.,
                 qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.blocks=nn.ModuleList([SwinTransformerBlock(
            dim,num_heads,window_size,0 if i%2==0 else window_size//2,
            mlp_ratio,qkv_bias,drop,attn_drop,
            drop_path[i] if isinstance(drop_path,list) else drop_path
        ) for i in range(depth)])
    def forward(self, x):
        for blk in self.blocks: x=blk(x)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=1, in_chans=3, embed_dim=96):
        super().__init__()
        self.proj=nn.Conv2d(in_chans,embed_dim,patch_size,patch_size); self.norm=nn.LayerNorm(embed_dim)
    def forward(self, x): return self.norm(self.proj(x).flatten(2).transpose(1,2))

class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=1, embed_dim=96, out_chans=3):
        super().__init__()
        self.proj=nn.ConvTranspose2d(embed_dim,out_chans,patch_size,patch_size)
    def forward(self, x, hw):
        H,W=hw; B,N,C=x.shape; return self.proj(x.transpose(1,2).view(B,C,H,W))


# ── FSCA v2 핵심: 채널 수준 Cross-Modulation ─────────────────
class FreqBranch(nn.Module):
    """FFT 기반 주파수 특징 추출 — float32 명시로 ComplexHalf 경고 방지"""
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim*2, dim*2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim*2, dim*2, 1)
        )
    def forward(self, x):
        # float32로 변환 후 FFT (AMP 환경에서 ComplexHalf 방지)
        x_f = x.float()
        fft = torch.fft.rfft2(x_f, norm='ortho')
        fft_feat = self.conv(torch.cat([fft.real, fft.imag], dim=1).to(x.dtype))
        fft_out = torch.complex(fft_feat[:,:x.shape[1]].float(), fft_feat[:,x.shape[1]:].float())
        return torch.fft.irfft2(fft_out, s=(x.shape[2], x.shape[3]), norm='ortho').to(x.dtype)


class FreqSpatialCrossModulation(nn.Module):
    """
    FSCA v2: FiLM(Feature-wise Linear Modulation) 기반 주파수→공간 변조
    전체 시퀀스 cross-attention 대신 글로벌 채널 컨텍스트 사용
    메모리: O(C²) — 훨씬 효율적
    
    동작:
    1. 주파수 특징 → Global Avg Pooling → 채널 컨텍스트 벡터
    2. 채널 벡터 → MLP → (gamma, beta) 생성
    3. 공간 특징 = gamma * 공간 특징 + beta  (FiLM 변조)
    4. 학습 가능한 게이트로 반영 강도 조절
    """
    def __init__(self, dim):
        super().__init__()
        # 주파수 글로벌 컨텍스트 → gamma, beta
        self.film_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),           # (B, C, 1, 1)
            nn.Flatten(1),                     # (B, C)
            nn.Linear(dim, dim*2),
            nn.GELU(),
            nn.Linear(dim*2, dim*2)            # → gamma(C) + beta(C)
        )
        self.norm = nn.LayerNorm(dim)
        # 게이트: 변조 강도 조절
        self.gate = nn.Parameter(torch.zeros(1))  # 학습 초기엔 0 (안전한 시작)

    def forward(self, spatial, freq):
        """
        spatial: (B, H, W, C)
        freq:    (B, C, H, W)
        반환:    (B, H, W, C)
        """
        B, H, W, C = spatial.shape

        # 주파수에서 채널 컨텍스트 추출 → gamma, beta
        film_params = self.film_gen(freq)        # (B, 2C)
        gamma = film_params[:, :C].view(B,1,1,C) + 1.0   # 초기값 1 (identity)
        beta  = film_params[:, C:].view(B,1,1,C)          # 초기값 0

        # FiLM 변조
        sp_norm = self.norm(spatial)
        modulated = gamma * sp_norm + beta

        # 게이트로 반영 강도 조절 (tanh로 안정화)
        return spatial + torch.tanh(self.gate) * modulated


class DehazeFormerFSCA(nn.Module):
    """
    DehazeFormer + FSCA v2
    Bottleneck 출력에 FreqBranch + FreqSpatialCrossModulation 삽입
    추가 파라미터: ~0.1M (기존 6.11M 대비 경량)
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
        cfg=self.CONFIGS[size]; embed_dim=cfg['embed_dim']
        depths=cfg['depths']; num_heads=cfg['num_heads']
        bn_dim=embed_dim*4

        self.patch_embed=PatchEmbed(1,in_chans,embed_dim)
        dpr=[x.item() for x in torch.linspace(0,drop_path_rate,sum(depths))]

        self.encoder1=BasicLayer(embed_dim,depths[0],num_heads[0],window_size,mlp_ratio,drop=drop_rate,attn_drop=attn_drop_rate,drop_path=dpr[:depths[0]])
        self.down1=nn.Conv2d(embed_dim,embed_dim*2,2,2)
        self.encoder2=BasicLayer(embed_dim*2,depths[1],num_heads[1],window_size,mlp_ratio,drop=drop_rate,attn_drop=attn_drop_rate,drop_path=dpr[depths[0]:sum(depths[:2])])
        self.down2=nn.Conv2d(embed_dim*2,bn_dim,2,2)

        self.bottleneck=BasicLayer(bn_dim,depths[2],num_heads[2],window_size,mlp_ratio,drop=drop_rate,attn_drop=attn_drop_rate,drop_path=dpr[sum(depths[:2]):sum(depths[:3])])

        # ── FSCA 모듈 ──
        self.freq_branch=FreqBranch(bn_dim)
        self.fsca=FreqSpatialCrossModulation(bn_dim)

        self.up2=nn.ConvTranspose2d(bn_dim,embed_dim*2,2,2)
        self.fusion2=SKFusion(embed_dim*2)
        self.decoder2=BasicLayer(embed_dim*2,depths[1],num_heads[1],window_size,mlp_ratio,drop=drop_rate,attn_drop=attn_drop_rate,drop_path=dpr[sum(depths[:1]):sum(depths[:2])])
        self.up1=nn.ConvTranspose2d(embed_dim*2,embed_dim,2,2)
        self.fusion1=SKFusion(embed_dim)
        self.decoder1=BasicLayer(embed_dim,depths[0],num_heads[0],window_size,mlp_ratio,drop=drop_rate,attn_drop=attn_drop_rate,drop_path=dpr[:depths[0]])

        self.patch_unembed=PatchUnEmbed(1,embed_dim,in_chans)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m,nn.Linear): trunc_normal_(m.weight,std=.02); nn.init.zeros_(m.bias) if m.bias is not None else None
        elif isinstance(m,(nn.LayerNorm,nn.BatchNorm2d)): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        B,C,H,W=x.shape
        feat=self.patch_embed(x).view(B,H,W,-1)

        e1=self.encoder1(feat); e1_c=e1.permute(0,3,1,2)
        e2=self.encoder2(self.down1(e1_c).permute(0,2,3,1)); e2_c=e2.permute(0,3,1,2)

        b=self.bottleneck(self.down2(e2_c).permute(0,2,3,1))  # (B,H4,W4,4C)
        b_c=b.permute(0,3,1,2)                                  # (B,4C,H4,W4)

        # FSCA: 주파수 특징 추출 → 공간 변조
        freq_feat=self.freq_branch(b_c)       # (B,4C,H4,W4)
        b_enhanced=self.fsca(b, freq_feat)    # (B,H4,W4,4C)
        b_c=b_enhanced.permute(0,3,1,2)

        d2=self.decoder2(self.fusion2([self.up2(b_c),e2_c]).permute(0,2,3,1))
        d2_c=d2.permute(0,3,1,2)
        d1=self.decoder1(self.fusion1([self.up1(d2_c),e1_c]).permute(0,2,3,1))

        out=self.patch_unembed(d1.view(B,H*W,-1),(H,W))
        return out+x

def build_model_fsca(size='B', **kwargs):
    return DehazeFormerFSCA(size=size, **kwargs)
