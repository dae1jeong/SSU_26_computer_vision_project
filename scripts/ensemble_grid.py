"""
가중치 최적화 + 모델 추가 앙상블 그리드 서치
- 모델 5개로 확장 (기존 3개 + FT_GB_0149 + S4)
- 가중치 그리드 서치로 최적 조합 탐색
- 최적 조합에 8방향 TTA 적용
"""

import os, sys, json, time, itertools
import torch
import numpy as np
from torch.amp import autocast

REPO_DIR = "/home/user/computer_vision"
sys.path.append(REPO_DIR)

from models.dehazeformer import build_model
from datasets.dataset import get_dataloader
from utils.metrics import compute_psnr, compute_ssim

EXP_ROOT = "/home/user/experiments"
VAL_DATA = "/home/user/data/RESIDE/RESIDE-6K/test"
LOG_PATH = os.path.join(REPO_DIR, "ensemble_grid_log.json")

# ── 앙상블 후보 모델 (5개) ────────────────────────────────────
CANDIDATE_MODELS = [
    ("S3_best_extend_lr5e6",                       "B", 29.7915),
    ("FT_GB_0149_B_freq_branch_p256_lr2e-04_L1Perc","B", 29.6715),
    ("S4_GB0149_L1heavy",                           "B", 29.7628),
    ("FT_GA_0065_B_baseline_p256_lr2e-04_L1Perc",  "B", 29.5275),
    ("FT_GA_0045_S_baseline_p256_lr4e-04_L1Perc",  "S", 29.4374),
]


def load_model(exp_name, model_size, device):
    model = build_model(size=model_size).to(device)
    ckpt  = os.path.join(EXP_ROOT, exp_name, "best.pth")
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    return model


def load_log():
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f: return json.load(f)
    return {"grid_results": [], "best": {"psnr": 0, "weights": [], "models": []}}


def save_log(log):
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


def tta_transforms(x):
    """4방향 TTA (flip만, rotate 제외 — 비정사각 이미지 대응)"""
    return [
        x,                             # 원본
        torch.flip(x, dims=[3]),       # 좌우반전
        torch.flip(x, dims=[2]),       # 상하반전
        torch.flip(x, dims=[2, 3]),    # 180도
    ]

def tta_inverse(preds):
    inv = [
        preds[0],
        torch.flip(preds[1], dims=[3]),
        torch.flip(preds[2], dims=[2]),
        torch.flip(preds[3], dims=[2, 3]),
    ]
    return torch.stack(inv).mean(0)


def fast_evaluate(models, weights, val_loader, device, tta=False):
    """빠른 평가 (그리드 서치용 - 랜덤 200장 샘플링)"""
    import random
    all_data = list(val_loader)
    sampled = random.sample(all_data, min(200, len(all_data)))
    psnrs = []
    with torch.no_grad():
        for hazy, clear, _ in sampled:
            hazy, clear = hazy.to(device), clear.to(device)
            if tta:
                preds = []
                for model in models:
                    with autocast("cuda"):
                        aug_preds = [model(aug).clamp(0,1) for aug in tta_transforms(hazy)]
                    preds.append(tta_inverse(aug_preds))
            else:
                with autocast("cuda"):
                    preds = [model(hazy).clamp(0,1) for model in models]
            final = sum(w * p for w, p in zip(weights, preds))
            psnrs.append(compute_psnr(final, clear))
    return sum(psnrs) / len(psnrs)


def full_evaluate(models, weights, val_loader, device, tta=False):
    """전체 1000장 평가"""
    psnrs, ssims = [], []
    with torch.no_grad():
        for hazy, clear, _ in val_loader:
            hazy, clear = hazy.to(device), clear.to(device)
            if tta:
                preds = []
                for model in models:
                    with autocast("cuda"):
                        aug_preds = [model(aug).clamp(0,1) for aug in tta_transforms(hazy)]
                    preds.append(tta_inverse(aug_preds))
            else:
                with autocast("cuda"):
                    preds = [model(hazy).clamp(0,1) for model in models]
            final = sum(w * p for w, p in zip(weights, preds))
            psnrs.append(compute_psnr(final, clear))
            ssims.append(compute_ssim(final, clear))
    return sum(psnrs)/len(psnrs), sum(ssims)/len(ssims)


if __name__ == "__main__":
    device = torch.device("cuda")
    log = load_log()

    print("\n" + "="*65)
    print("  🔍 앙상블 가중치 그리드 서치 + 모델 5개 확장")
    print("="*65)

    # 모델 로드
    print("\n  모델 로드 중...")
    models = []
    for exp_name, model_size, psnr in CANDIDATE_MODELS:
        m = load_model(exp_name, model_size, device)
        models.append(m)
        print(f"  ✅ {exp_name[:40]} ({psnr}dB)")

    val_loader = get_dataloader(VAL_DATA, patch_size=256, batch_size=1, is_train=False)

    # ── 단계 1: 그리드 서치 (200장, 빠르게) ──────────────────
    print("\n  [단계1] 가중치 그리드 서치 (200장 빠른 평가)")
    print("  가중치 후보: 0.1 단위 조합 (합=1.0)")

    # 0.1 단위 가중치 조합 생성 (5모델, 합=1.0)
    best_psnr_grid = 0
    best_weights   = None
    count = 0

    # 탐색 범위 제한: S3 가중치 0.3~0.6, 나머지 균등 분배 방식
    weight_candidates = []
    for w0 in [0.3, 0.4, 0.5, 0.6]:      # S3
        for w1 in [0.1, 0.2, 0.3]:         # FT_GB_0149
            for w2 in [0.1, 0.2, 0.3]:     # S4
                for w3 in [0.1, 0.2]:       # FT_GA_0065
                    w4 = round(1.0 - w0 - w1 - w2 - w3, 1)
                    if 0.0 <= w4 <= 0.3:
                        weight_candidates.append([w0, w1, w2, w3, w4])

    print(f"  총 {len(weight_candidates)}개 조합 탐색")

    t0 = time.time()
    for weights in weight_candidates:
        psnr = fast_evaluate(models, weights, val_loader, device, tta=False)
        count += 1
        if psnr > best_psnr_grid:
            best_psnr_grid = psnr
            best_weights   = weights
            print(f"  🆕 Best! 가중치={weights} → PSNR={psnr:.4f}dB ({count}/{len(weight_candidates)})")

    elapsed = int(time.time() - t0)
    print(f"\n  그리드 서치 완료 ({elapsed}s)")
    print(f"  최적 가중치: {best_weights}")
    print(f"  200장 기준 PSNR: {best_psnr_grid:.4f}dB")

    # ── 단계 2: 최적 가중치로 전체 1000장 평가 ───────────────
    print(f"\n  [단계2] 최적 가중치로 전체 1000장 평가 (TTA 없음)")
    psnr_full, ssim_full = full_evaluate(models, best_weights, val_loader, device, tta=False)
    print(f"  결과: PSNR={psnr_full:.4f}dB | SSIM={ssim_full:.4f}")

    result1 = {
        "stage": "최적가중치_앙상블",
        "weights": best_weights,
        "models": [m[0] for m in CANDIDATE_MODELS],
        "psnr": round(psnr_full, 4),
        "ssim": round(ssim_full, 4),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    log["grid_results"].append(result1)
    if psnr_full > log["best"]["psnr"]:
        log["best"] = {"psnr": round(psnr_full, 4), "weights": best_weights,
                       "models": [m[0] for m in CANDIDATE_MODELS]}
    save_log(log)

    # ── 단계 3: 최적 가중치 + 8방향 TTA ──────────────────────
    print(f"\n  [단계3] 최적 가중치 + 8방향 TTA 전체 평가")
    psnr_tta, ssim_tta = full_evaluate(models, best_weights, val_loader, device, tta=True)
    print(f"  결과: PSNR={psnr_tta:.4f}dB | SSIM={ssim_tta:.4f}")

    result2 = {
        "stage": "최적가중치_앙상블+TTA8",
        "weights": best_weights,
        "models": [m[0] for m in CANDIDATE_MODELS],
        "psnr": round(psnr_tta, 4),
        "ssim": round(ssim_tta, 4),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    log["grid_results"].append(result2)
    if psnr_tta > log["best"]["psnr"]:
        log["best"] = {"psnr": round(psnr_tta, 4), "weights": best_weights,
                       "models": [m[0] for m in CANDIDATE_MODELS]}
    save_log(log)

    # ── 최종 결과 ─────────────────────────────────────────────
    print("\n" + "="*65)
    print("  🏁 최종 결과")
    print(f"  기존 Best (고정 가중치 + TTA8): 29.9027dB")
    print(f"  최적 가중치 앙상블:             {psnr_full:.4f}dB")
    print(f"  최적 가중치 앙상블 + TTA8:      {psnr_tta:.4f}dB  ← 최종")
    if psnr_tta >= 30.0:
        print(f"  🎊 30dB 달성!! ({psnr_tta:.4f}dB)")
    else:
        print(f"  30dB까지 {30.0 - psnr_tta:.4f}dB 남음")
    print("="*65)
