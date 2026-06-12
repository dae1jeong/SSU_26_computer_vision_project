"""
앙상블 + TTA + Self-Ensemble 순차 평가 스크립트
단계 1: 가중 앙상블 (S3 + FT_GA_0065 + FT_GA_0045)
단계 2: 앙상블 + TTA (좌우반전)
단계 3: 앙상블 + Self-Ensemble (8방향 TTA)
"""

import os, sys, json, time
import torch
import torch.nn as nn
import numpy as np
from torch.amp import autocast

REPO_DIR = "/home/user/computer_vision"
sys.path.append(REPO_DIR)

from models.dehazeformer import build_model
from datasets.dataset import get_dataloader
from utils.metrics import compute_psnr, compute_ssim

EXP_ROOT  = "/home/user/experiments"
VAL_DATA  = "/home/user/data/RESIDE/RESIDE-6K/test"
LOG_PATH  = os.path.join(REPO_DIR, "ensemble_log.json")

# ── 앙상블 모델 정의 ──────────────────────────────────────────
ENSEMBLE_MODELS = [
    # (실험명, 모델크기, PSNR, 가중치)
    ("S3_best_extend_lr5e6",                          "B", 29.7915, 0.5),
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


def load_log():
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            return json.load(f)
    return {"results": [], "best": {"psnr": 0, "method": ""}}


def save_log(log):
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


# ── TTA 변환 함수들 ──────────────────────────────────────────
def tta_transforms(x):
    """4방향 TTA (비정사각 이미지 대응, rotate 제외)"""
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


# ── 평가 함수 ─────────────────────────────────────────────────
def evaluate(models, weights, val_loader, device, tta_mode="none"):
    """
    tta_mode: 'none' | 'flip' | 'self8'
    """
    psnrs, ssims = [], []
    weights_t = torch.tensor(weights, device=device)

    with torch.no_grad():
        for hazy, clear, _ in val_loader:
            hazy, clear = hazy.to(device), clear.to(device)

            if tta_mode == "none":
                # 단순 가중 앙상블
                preds = []
                for model in models:
                    with autocast("cuda"):
                        pred = model(hazy).clamp(0, 1)
                    preds.append(pred)
                final = sum(w * p for w, p in zip(weights, preds))

            elif tta_mode == "flip":
                # 좌우반전 TTA + 가중 앙상블
                preds = []
                for model in models:
                    with autocast("cuda"):
                        p1 = model(hazy).clamp(0, 1)
                        p2 = torch.flip(model(torch.flip(hazy, dims=[3])).clamp(0, 1), dims=[3])
                    preds.append((p1 + p2) / 2)
                final = sum(w * p for w, p in zip(weights, preds))

            elif tta_mode == "self8":
                # 8방향 Self-Ensemble + 가중 앙상블
                preds = []
                for model in models:
                    aug_preds = []
                    for aug in tta_transforms(hazy):
                        with autocast("cuda"):
                            aug_pred = model(aug).clamp(0, 1)
                        aug_preds.append(aug_pred)
                    preds.append(tta_inverse(aug_preds))
                final = sum(w * p for w, p in zip(weights, preds))

            psnrs.append(compute_psnr(final, clear))
            ssims.append(compute_ssim(final, clear))

    return sum(psnrs)/len(psnrs), sum(ssims)/len(ssims)


# ── 메인 ──────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda")
    log = load_log()

    print("\n" + "="*65)
    print("  🎯 앙상블 + TTA 평가 시작")
    print(f"  모델: {[m[0][:20] for m in ENSEMBLE_MODELS]}")
    print(f"  가중치: {[m[3] for m in ENSEMBLE_MODELS]}")
    print("="*65)

    # 모델 로드
    print("\n  모델 로드 중...")
    models, weights = [], []
    for exp_name, model_size, psnr, w in ENSEMBLE_MODELS:
        m = load_model(exp_name, model_size, device)
        models.append(m)
        weights.append(w)
        print(f"  ✅ {exp_name[:35]} ({psnr}dB, w={w})")

    # 데이터로더
    val_loader = get_dataloader(VAL_DATA, patch_size=256, batch_size=1, is_train=False)
    print(f"\n  Val 데이터: {VAL_DATA}")

    stages = [
        ("단계1_앙상블",          "none"),
        ("단계2_앙상블+TTA_flip", "flip"),
        ("단계3_앙상블+TTA_8방향","self8"),
    ]

    for stage_name, tta_mode in stages:
        # 이미 완료한 단계 스킵
        if any(r["stage"] == stage_name for r in log["results"]):
            prev = next(r for r in log["results"] if r["stage"] == stage_name)
            print(f"\n  ⏭️  {stage_name} 이미 완료 → {prev['psnr']:.4f}dB")
            continue

        print(f"\n  {'='*55}")
        print(f"  🔄 {stage_name} 평가 중... (tta={tta_mode})")
        t0 = time.time()
        psnr, ssim = evaluate(models, weights, val_loader, device, tta_mode)
        elapsed = int(time.time() - t0)

        result = {
            "stage": stage_name,
            "tta_mode": tta_mode,
            "psnr": round(psnr, 4),
            "ssim": round(ssim, 4),
            "elapsed_sec": elapsed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        log["results"].append(result)

        if psnr > log["best"]["psnr"]:
            log["best"] = {"psnr": round(psnr, 4), "method": stage_name}

        save_log(log)

        star = " 🌟 30dB 달성!!" if psnr >= 30.0 else ""
        print(f"  ✅ {stage_name}: PSNR={psnr:.4f}dB | SSIM={ssim:.4f} | {elapsed}s{star}")

    print("\n" + "="*65)
    print("  🏁 전체 평가 완료")
    for r in log["results"]:
        tag = " 👑" if r["psnr"] == log["best"]["psnr"] else ""
        print(f"  {r['stage']}: {r['psnr']:.4f}dB{tag}")
    print(f"\n  🏆 Best: {log['best']['method']} → {log['best']['psnr']:.4f}dB")
    if log["best"]["psnr"] >= 30.0:
        print("  🎊 30dB 달성!!")
    else:
        print(f"  30dB까지 {30.0 - log['best']['psnr']:.4f}dB 남음")
    print("="*65)
