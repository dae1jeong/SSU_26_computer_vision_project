"""
PPT용 시각 비교 이미지 생성
Before (안개) / After (디헤이징) / After+ (후처리) 3단 비교
S7 앙상블 모델 사용
"""
import os, sys
import torch
import torch.nn.functional as F
from torch.amp import autocast
from PIL import Image, ImageEnhance, ImageFilter
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.append('/home/user/computer_vision')
from models.dehazeformer import build_model
from utils.metrics import compute_psnr, compute_ssim

EXP_ROOT  = '/home/user/experiments'
HAZY_DIR  = '/home/user/data/RESIDE/RESIDE-6K/test/hazy'
GT_DIR    = '/home/user/data/RESIDE/RESIDE-6K/test/GT'
OUT_DIR   = '/home/user/computer_vision/visual_comparison'
os.makedirs(OUT_DIR, exist_ok=True)

# S7 Best 앙상블 가중치
ENSEMBLE_CFG = [
    ('S5_charbonnier_cyclic',                       'B', 0.6),
    ('FT_GB_0149_B_freq_branch_p256_lr2e-04_L1Perc','B', 0.1),
    ('FT_GA_0065_B_baseline_p256_lr2e-04_L1Perc',  'B', 0.21),
    ('FT_GA_0045_S_baseline_p256_lr4e-04_L1Perc',  'S', 0.09),
]

# 선별 이미지 (다양한 장면)
TARGET_IMAGES = [
    '0258_0.8_0.2.jpg',   # 548x412 - 안개 짙음
    '0287_0.8_0.2.jpg',   # 548x412
    '0239_0.8_0.2.jpg',   # 548x412
    '0253_0.8_0.2.jpg',   # 548x412
    '1781_0.8_0.2.jpg',   # 548x412
    '0179_0.8_0.2.jpg',   # 548x412
]

def load_models(device):
    models, weights = [], []
    for exp_name, size, w in ENSEMBLE_CFG:
        m = build_model(size=size).to(device)
        state = torch.load(f'{EXP_ROOT}/{exp_name}/best.pth', map_location=device, weights_only=False)
        m.load_state_dict(state['model'] if 'model' in state else state)
        m.eval()
        models.append(m)
        weights.append(w)
    print(f'  모델 {len(models)}개 로드 완료')
    return models, weights

def infer_ensemble_tta(models, weights, x, device, align=32):
    """S7 앙상블 + TTA 4방향"""
    B, C, H, W = x.shape
    pad_h = (align - H % align) % align
    pad_w = (align - W % align) % align
    
    augs = [x, torch.flip(x,[3]), torch.flip(x,[2]), torch.flip(x,[2,3])]
    final = None
    with torch.no_grad():
        for m, w in zip(models, weights):
            s = None
            for k, aug in enumerate(augs):
                xp = F.pad(aug, (0, pad_w, 0, pad_h), mode='reflect')
                with autocast('cuda'):
                    out = m(xp).clamp(0,1)[:,:,:H,:W]
                if k==1: out=torch.flip(out,[3])
                elif k==2: out=torch.flip(out,[2])
                elif k==3: out=torch.flip(out,[2,3])
                s = out if s is None else s+out
            final = (s/4)*w if final is None else final+(s/4)*w
    return final.clamp(0,1)

def postprocess(img_pil):
    """후처리: contrast + saturation + 약한 sharpening"""
    # Contrast 향상
    img = ImageEnhance.Contrast(img_pil).enhance(1.15)
    # Saturation 향상 (색감 살리기)
    img = ImageEnhance.Color(img).enhance(1.2)
    # 약한 sharpening
    img = img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=3))
    return img

def tensor_to_pil(t):
    arr = t.squeeze(0).cpu().numpy().transpose(1,2,0)
    arr = (arr * 255).clip(0,255).astype(np.uint8)
    return Image.fromarray(arr)

def make_comparison(fname, models, weights, device, idx):
    hazy_path = os.path.join(HAZY_DIR, fname)
    gt_path   = os.path.join(GT_DIR,   fname)
    
    hazy_pil = Image.open(hazy_path).convert('RGB')
    gt_pil   = Image.open(gt_path).convert('RGB')
    
    # 텐서 변환
    hazy_t = TF.to_tensor(hazy_pil).unsqueeze(0).to(device)
    gt_t   = TF.to_tensor(gt_pil).unsqueeze(0).to(device)
    
    # 추론
    dehazed_t = infer_ensemble_tta(models, weights, hazy_t, device)
    
    # PSNR/SSIM 계산
    # GT와 같은 크기로 맞추기
    H_gt, W_gt = gt_t.shape[2], gt_t.shape[3]
    dehazed_crop = dehazed_t[:,:,:H_gt,:W_gt] if dehazed_t.shape[2] >= H_gt else dehazed_t
    psnr_val = compute_psnr(dehazed_crop, gt_t[:,:,:dehazed_crop.shape[2],:dehazed_crop.shape[3]])
    
    # PIL 변환
    dehazed_pil = tensor_to_pil(dehazed_t)
    post_pil    = postprocess(dehazed_pil)
    
    W, H = hazy_pil.size
    
    # ── 3단 비교 이미지 생성 ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('#1a1a2e')
    
    titles = ['Before (Hazy Input)', 'After (DehazeFormer+Ensemble)', 'After+ (Post-processed)']
    images = [hazy_pil, dehazed_pil, post_pil]
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    
    for ax, img, title, color in zip(axes, images, titles, colors):
        ax.imshow(img)
        ax.set_title(title, fontsize=14, color='white', fontweight='bold', pad=10)
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
            spine.set_visible(True)
    
    # PSNR 표시
    fig.text(0.5, 0.02, f'PSNR: {psnr_val:.2f} dB  |  Model: S7 Ensemble (S5+GB_0149+GA_0065+GA_0045) + TTA  |  Final Best: 30.00 dB',
             ha='center', color='#aaaaaa', fontsize=11)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out_path = os.path.join(OUT_DIR, f'comparison_{idx:02d}_{fname.replace(".jpg","")}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    
    # 개별 이미지도 저장 (PPT 자유 배치용)
    hazy_pil.save(os.path.join(OUT_DIR, f'{idx:02d}_hazy_{fname}'))
    dehazed_pil.save(os.path.join(OUT_DIR, f'{idx:02d}_dehazed_{fname}'))
    post_pil.save(os.path.join(OUT_DIR, f'{idx:02d}_postprocessed_{fname}'))
    
    print(f'  [{idx}] {fname} → PSNR={psnr_val:.2f}dB | 저장: comparison_{idx:02d}_*.png')
    return psnr_val


if __name__ == '__main__':
    device = torch.device('cuda')
    print('\n' + '='*60)
    print('  PPT용 시각 비교 이미지 생성')
    print('  S7 앙상블 + TTA + 후처리')
    print('='*60)
    
    models, weights = load_models(device)
    
    results = []
    for i, fname in enumerate(TARGET_IMAGES, 1):
        hazy_path = os.path.join(HAZY_DIR, fname)
        gt_path   = os.path.join(GT_DIR, fname)
        if not os.path.exists(hazy_path) or not os.path.exists(gt_path):
            print(f'  [{i}] {fname} - 파일 없음, 스킵')
            continue
        psnr = make_comparison(fname, models, weights, device, i)
        results.append((fname, psnr))
    
    print('\n' + '='*60)
    print(f'  완료! {len(results)}개 이미지 생성')
    print(f'  저장 위치: {OUT_DIR}')
    for f, p in results:
        print(f'  {f}: {p:.2f}dB')
    print('='*60)
