"""
3단계: Sliding Window 앙상블 추론
256 패치를 stride=128로 슬라이딩 → 겹치는 영역 평균
경계 부근 품질 저하 보완 + S5 best 모델에 TTA 결합
"""

import os, sys, json, time
sys.path.append('/home/user/computer_vision')

import torch
import torch.nn.functional as F
from torch.amp import autocast
from models.dehazeformer import build_model
from datasets.dataset import get_dataloader
from utils.metrics import compute_psnr, compute_ssim

EXP_ROOT = "/home/user/experiments"
VAL_DATA = "/home/user/data/RESIDE/RESIDE-6K/test"
LOG_PATH = "/home/user/computer_vision/sliding_window_log.json"

PATCH  = 256
STRIDE = 128   # 50% overlap


def sliding_window_predict(model, hazy, patch=256, stride=128, device='cuda'):
    """슬라이딩 윈도우 추론 — 겹치는 영역은 평균"""
    B, C, H, W = hazy.shape
    output = torch.zeros_like(hazy)
    count  = torch.zeros(B, 1, H, W, device=device)

    for y in range(0, H - patch + 1, stride):
        for x in range(0, W - patch + 1, stride):
            patch_in = hazy[:, :, y:y+patch, x:x+patch]
            with autocast('cuda'):
                patch_out = model(patch_in).clamp(0, 1)
            output[:, :, y:y+patch, x:x+patch] += patch_out
            count[:, :, y:y+patch, x:x+patch]  += 1

    # 마지막 행/열 처리 (이미지 크기가 stride 배수가 아닌 경우)
    if H % stride != 0:
        y = H - patch
        for x in range(0, W - patch + 1, stride):
            patch_in  = hazy[:, :, y:y+patch, x:x+patch]
            with autocast('cuda'):
                patch_out = model(patch_in).clamp(0, 1)
            output[:, :, y:y+patch, x:x+patch] += patch_out
            count[:, :, y:y+patch, x:x+patch]  += 1
    if W % stride != 0:
        x = W - patch
        for y in range(0, H - patch + 1, stride):
            patch_in  = hazy[:, :, y:y+patch, x:x+patch]
            with autocast('cuda'):
                patch_out = model(patch_in).clamp(0, 1)
            output[:, :, y:y+patch, x:x+patch] += patch_out
            count[:, :, y:y+patch, x:x+patch]  += 1

    count = count.clamp(min=1)
    return (output / count).clamp(0, 1)


def flip_tta_sliding(model, hazy, patch, stride, device):
    """슬라이딩 윈도우 + 4방향 TTA"""
    flips = [
        hazy,
        torch.flip(hazy, dims=[3]),
        torch.flip(hazy, dims=[2]),
        torch.flip(hazy, dims=[2,3]),
    ]
    preds = []
    for i, x in enumerate(flips):
        p = sliding_window_predict(model, x, patch, stride, device)
        if i == 1: p = torch.flip(p, dims=[3])
        if i == 2: p = torch.flip(p, dims=[2])
        if i == 3: p = torch.flip(p, dims=[2,3])
        preds.append(p)
    return torch.stack(preds).mean(0)


def load_model(exp_name, size, device):
    model = build_model(size=size).to(device)
    ckpt  = f"{EXP_ROOT}/{exp_name}/best.pth"
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state['model'] if 'model' in state else state)
    model.eval()
    return model


if __name__ == "__main__":
    device = torch.device('cuda')
    log = {"results": [], "best": {"psnr": 0, "method": ""}}

    print("\n" + "="*65)
    print("  🪟 Sliding Window 앙상블 추론")
    print(f"  patch={PATCH}, stride={STRIDE} (50% overlap)")
    print("="*65)

    val_loader = get_dataloader(VAL_DATA, patch_size=PATCH, batch_size=1, is_train=False)

    # 평가할 모델 목록 (S5 완료 후 추가될 수도 있음)
    eval_targets = [
        ("S3_best_extend_lr5e6",  "B", "S3"),
    ]
    # S5 완료됐으면 추가
    s5_path = f"{EXP_ROOT}/S5_charbonnier_cyclic/best.pth"
    if os.path.exists(s5_path):
        eval_targets.append(("S5_charbonnier_cyclic", "B", "S5"))
        print("  ✅ S5 체크포인트 발견 — S5도 평가")
    else:
        print("  ℹ️  S5 아직 미완료 — S3만 평가")

    for exp_name, size, tag in eval_targets:
        print(f"\n  [{tag}] 모델 로드: {exp_name}")
        model = load_model(exp_name, size, device)

        # 방법 1: 슬라이딩 윈도우만
        print(f"  {tag} 슬라이딩 윈도우 평가 중...")
        t0 = time.time()
        psnrs, ssims = [], []
        with torch.no_grad():
            for hazy, clear, _ in val_loader:
                hazy, clear = hazy.to(device), clear.to(device)
                pred = sliding_window_predict(model, hazy, PATCH, STRIDE, device)
                psnrs.append(compute_psnr(pred, clear))
                ssims.append(compute_ssim(pred, clear))
        psnr1 = sum(psnrs)/len(psnrs)
        ssim1 = sum(ssims)/len(ssims)
        t1 = int(time.time()-t0)
        print(f"  {tag} 슬라이딩: PSNR={psnr1:.4f}dB | SSIM={ssim1:.4f} | {t1}s")

        r1 = {"method": f"{tag}_sliding", "psnr": round(psnr1,4),
              "ssim": round(ssim1,4), "elapsed": t1,
              "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        log["results"].append(r1)
        if psnr1 > log["best"]["psnr"]:
            log["best"] = {"psnr": round(psnr1,4), "method": f"{tag}_sliding"}

        # 방법 2: 슬라이딩 윈도우 + TTA
        print(f"  {tag} 슬라이딩 + TTA 평가 중...")
        t0 = time.time()
        psnrs, ssims = [], []
        with torch.no_grad():
            for hazy, clear, _ in val_loader:
                hazy, clear = hazy.to(device), clear.to(device)
                pred = flip_tta_sliding(model, hazy, PATCH, STRIDE, device)
                psnrs.append(compute_psnr(pred, clear))
                ssims.append(compute_ssim(pred, clear))
        psnr2 = sum(psnrs)/len(psnrs)
        ssim2 = sum(ssims)/len(ssims)
        t2 = int(time.time()-t0)
        print(f"  {tag} 슬라이딩+TTA: PSNR={psnr2:.4f}dB | SSIM={ssim2:.4f} | {t2}s")
        if psnr2 >= 30.0:
            print(f"  🎊 30dB 달성!! ({psnr2:.4f}dB)")

        r2 = {"method": f"{tag}_sliding_tta", "psnr": round(psnr2,4),
              "ssim": round(ssim2,4), "elapsed": t2,
              "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        log["results"].append(r2)
        if psnr2 > log["best"]["psnr"]:
            log["best"] = {"psnr": round(psnr2,4), "method": f"{tag}_sliding_tta"}

    with open(LOG_PATH, 'w') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    print("\n" + "="*65)
    print("  🏁 Sliding Window 평가 완료")
    for r in log["results"]:
        tag = " 👑" if r["psnr"] == log["best"]["psnr"] else ""
        print(f"  {r['method']}: {r['psnr']:.4f}dB{tag}")
    print(f"\n  🏆 Best: {log['best']['method']} → {log['best']['psnr']:.4f}dB")
    print("="*65)
