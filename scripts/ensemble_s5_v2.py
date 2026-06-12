"""
S5 앙상블 + TTA 평가 (수정본)
"""
import os, sys, json, time
import torch

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
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    return model

def run_ensemble(models, weights, val_loader, device, tta=False):
    psnrs, ssims = [], []
    with torch.no_grad():
        for hazy, clear, _ in val_loader:
            hazy  = hazy.to(device)
            clear = clear.to(device)

            if tta:
                # 4방향 flip
                augs = [
                    hazy,
                    torch.flip(hazy, [3]),
                    torch.flip(hazy, [2]),
                    torch.flip(hazy, [2,3]),
                ]
                # 각 모델, 각 aug 순전파 → 역변환 → 평균
                final_out = None
                for m, w in zip(models, weights):
                    model_out_sum = None
                    for k, aug in enumerate(augs):
                        with torch.amp.autocast("cuda"):
                            out = m(aug).clamp(0, 1)
                        # 역변환
                        if k == 1: out = torch.flip(out, [3])
                        elif k == 2: out = torch.flip(out, [2])
                        elif k == 3: out = torch.flip(out, [2,3])
                        model_out_sum = out if model_out_sum is None else model_out_sum + out
                    model_out_sum = model_out_sum / len(augs)
                    final_out = model_out_sum * w if final_out is None else final_out + model_out_sum * w
            else:
                final_out = None
                for m, w in zip(models, weights):
                    with torch.amp.autocast("cuda"):
                        out = m(hazy).clamp(0, 1)
                    final_out = out * w if final_out is None else final_out + out * w

            final_out = final_out.clamp(0, 1)
            psnrs.append(compute_psnr(final_out, clear))
            ssims.append(compute_ssim(final_out, clear))

    return sum(psnrs)/len(psnrs), sum(ssims)/len(ssims)

def main():
    device = torch.device("cuda")
    print("="*60)
    print("  S5 포함 앙상블 + TTA 평가 (v2)")
    print("="*60)

    models, weights = [], []
    for exp_name, size, psnr, w in ENSEMBLE_MODELS:
        print(f"  로드: {exp_name} (w={w})")
        models.append(load_model(exp_name, size, device))
        weights.append(w)

    val_loader = get_dataloader(VAL_DATA, batch_size=1, patch_size=256, is_train=False)

    print("\n  [단계1] 앙상블 (no TTA)...")
    t0 = time.time()
    p1, s1 = run_ensemble(models, weights, val_loader, device, tta=False)
    print(f"  앙상블: PSNR={p1:.4f}dB | SSIM={s1:.4f} | {int(time.time()-t0)}s")

    print("  [단계2] 앙상블 + TTA 4방향...")
    t0 = time.time()
    p2, s2 = run_ensemble(models, weights, val_loader, device, tta=True)
    print(f"  앙상블+TTA: PSNR={p2:.4f}dB | SSIM={s2:.4f} | {int(time.time()-t0)}s")

    best = max(p1, p2)
    best_m = "앙상블+TTA" if p2 >= p1 else "앙상블"
    print()
    print("="*60)
    print(f"  🏆 Best: {best_m} → {best:.4f}dB")
    print(f"  30dB 까지: {30.0-best:+.4f}dB")
    if best >= 30.0:
        print("  🎉🎉 30dB 달성!! 🎉🎉")
    print("="*60)

    log = {"no_tta": {"psnr": p1, "ssim": s1},
           "tta":    {"psnr": p2, "ssim": s2},
           "best_psnr": best, "best_method": best_m}
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)

if __name__ == "__main__":
    main()
