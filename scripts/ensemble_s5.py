"""
S5 포함 앙상블 + TTA 평가
모델: S5_charbonnier_cyclic (w=0.5) + FT_GA_0065 (w=0.3) + FT_GA_0045 (w=0.2)
"""
import os, sys, json, time
import torch
import numpy as np

REPO_DIR = "/home/user/computer_vision"
sys.path.append(REPO_DIR)

from models.dehazeformer import build_model
from datasets.dataset import get_dataloader
from utils.metrics import compute_psnr, compute_ssim

EXP_ROOT = "/home/user/experiments"
VAL_DATA = "/home/user/data/RESIDE/RESIDE-6K/test"
LOG_PATH = os.path.join(REPO_DIR, "ensemble_s5_log.json")

ENSEMBLE_MODELS = [
    ("S5_charbonnier_cyclic",                         "B", 29.9096, 0.5),
    ("FT_GA_0065_B_baseline_p256_lr2e-04_L1Perc",    "B", 29.5275, 0.3),
    ("FT_GA_0045_S_baseline_p256_lr4e-04_L1Perc",    "S", 29.4374, 0.2),
]

def load_model(exp_name, model_size, device):
    model = build_model(size=model_size).to(device)
    ckpt_path = os.path.join(EXP_ROOT, exp_name, "best.pth")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model

def tta_transforms(x):
    return [
        x,
        torch.flip(x, dims=[3]),
        torch.flip(x, dims=[2]),
        torch.flip(x, dims=[2, 3]),
    ]

def tta_inverse(preds):
    inv = [
        preds[0],
        torch.flip(preds[1], dims=[3]),
        torch.flip(preds[2], dims=[2]),
        torch.flip(preds[3], dims=[2, 3]),
    ]
    return torch.stack(inv).mean(0)

def evaluate(models, weights, val_loader, device, tta=False):
    psnrs, ssims = [], []
    with torch.no_grad():
        for hazy, clear, _ in val_loader:
            hazy, clear = hazy.to(device), clear.to(device)
            if tta:
                aug_inputs = tta_transforms(hazy)
                batch_out = []
                for m, w in zip(models, weights):
                    model_preds = []
                    for aug in aug_inputs:
                        with torch.amp.autocast("cuda"):
                            out = m(aug)
                        model_preds.append(out)
                    batch_out.append(tta_inverse(model_preds) * w)
                pred = torch.stack(batch_out).sum(0)
            else:
                preds = []
                for m, w in zip(models, weights):
                    with torch.amp.autocast("cuda"):
                        out = m(hazy)
                    preds.append(out * w)
                pred = torch.stack(preds).sum(0)
            pred = pred.clamp(0, 1)
            psnrs.append(compute_psnr(pred, clear))
            ssims.append(compute_ssim(pred, clear))
    return sum(psnrs)/len(psnrs), sum(ssims)/len(ssims)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("="*60)
    print("  S5 포함 앙상블 + TTA 평가")
    print("="*60)

    # 모델 로드
    models, weights = [], []
    for exp_name, size, psnr, w in ENSEMBLE_MODELS:
        print(f"  로드 중: {exp_name} (size={size}, base={psnr}dB, w={w})")
        m = load_model(exp_name, size, device)
        models.append(m)
        weights.append(w)
    print()

    val_loader = get_dataloader(VAL_DATA, batch_size=1, patch_size=256, is_train=False)

    # 단계1: 앙상블만
    print("  [단계1] S5 앙상블 (no TTA) 평가 중...")
    t0 = time.time()
    psnr1, ssim1 = evaluate(models, weights, val_loader, device, tta=False)
    print(f"  S5 앙상블: PSNR={psnr1:.4f}dB | SSIM={ssim1:.4f} | {int(time.time()-t0)}s")

    # 단계2: 앙상블 + TTA
    print("  [단계2] S5 앙상블 + TTA 4방향 평가 중...")
    t0 = time.time()
    psnr2, ssim2 = evaluate(models, weights, val_loader, device, tta=True)
    print(f"  S5 앙상블+TTA: PSNR={psnr2:.4f}dB | SSIM={ssim2:.4f} | {int(time.time()-t0)}s")

    best_psnr = max(psnr1, psnr2)
    best_method = "S5_ensemble+TTA" if psnr2 >= psnr1 else "S5_ensemble"

    print()
    print("="*60)
    print(f"  🏆 Best: {best_method} → {best_psnr:.4f}dB")
    print(f"  30dB 까지: {30.0 - best_psnr:.4f}dB")
    print("="*60)

    log = {
        "ensemble_no_tta": {"psnr": psnr1, "ssim": ssim1},
        "ensemble_tta":     {"psnr": psnr2, "ssim": ssim2},
        "best_psnr": best_psnr,
        "best_method": best_method,
    }
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  로그 저장: {LOG_PATH}")

if __name__ == "__main__":
    main()
