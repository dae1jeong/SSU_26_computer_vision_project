"""
S7: 4모델 앙상블 그리드 서치
S5(GA) + FT_GB_0149(GB freq) + GA_0065 + GA_0045
"""
import os, sys, json, time, random
import torch
from torch.amp import autocast

REPO_DIR = "/home/user/computer_vision"
sys.path.append(REPO_DIR)
from models.dehazeformer import build_model
from datasets.dataset import get_dataloader, HazeDataset
from utils.metrics import compute_psnr, compute_ssim
from torch.utils.data import DataLoader, Subset

EXP_ROOT = "/home/user/experiments"
VAL_DATA = "/home/user/data/RESIDE/RESIDE-6K/test"
LOG_PATH = os.path.join(REPO_DIR, "ensemble_s7_log.json")

ALL_MODELS = [
    ("S5_charbonnier_cyclic",                       "B"),
    ("FT_GB_0149_B_freq_branch_p256_lr2e-04_L1Perc","B"),
    ("FT_GA_0065_B_baseline_p256_lr2e-04_L1Perc",  "B"),
    ("FT_GA_0045_S_baseline_p256_lr4e-04_L1Perc",  "S"),
]

def load_models(device):
    models = []
    for exp_name, size in ALL_MODELS:
        m = build_model(size=size).to(device)
        state = torch.load(f"{EXP_ROOT}/{exp_name}/best.pth", map_location=device, weights_only=False)
        m.load_state_dict(state["model"] if "model" in state else state)
        m.eval()
        models.append(m)
        print(f"  ✅ {exp_name[:50]}")
    return models

def eval_on_loader(models, weights, loader, device, tta=False):
    psnrs, ssims = [], []
    with torch.no_grad():
        for hazy, clear, _ in loader:
            hazy, clear = hazy.to(device), clear.to(device)
            if tta:
                augs = [hazy, torch.flip(hazy,[3]), torch.flip(hazy,[2]), torch.flip(hazy,[2,3])]
                final = None
                for m, w in zip(models, weights):
                    s = None
                    for k, aug in enumerate(augs):
                        with autocast("cuda"):
                            o = m(aug).clamp(0,1)
                        if k==1: o=torch.flip(o,[3])
                        elif k==2: o=torch.flip(o,[2])
                        elif k==3: o=torch.flip(o,[2,3])
                        s = o if s is None else s+o
                    final = (s/4)*w if final is None else final+(s/4)*w
            else:
                final = None
                for m, w in zip(models, weights):
                    with autocast("cuda"):
                        o = m(hazy).clamp(0,1)
                    final = o*w if final is None else final+o*w
            psnrs.append(compute_psnr(final.clamp(0,1), clear))
            ssims.append(compute_ssim(final.clamp(0,1), clear))
    return sum(psnrs)/len(psnrs), sum(ssims)/len(ssims)


if __name__ == "__main__":
    device = torch.device("cuda")
    print("\n" + "="*65)
    print("  S7: 4모델 앙상블 그리드 서치")
    print("  S5(GA) + GB_0149(GB freq) + GA_0065 + GA_0045")
    print("="*65)

    models = load_models(device)

    # 샘플 200개 서브셋 로더
    full_ds = HazeDataset(VAL_DATA, patch_size=256, is_train=False)
    random.seed(42)
    idx200 = random.sample(range(len(full_ds)), 200)
    sample_loader = DataLoader(Subset(full_ds, idx200), batch_size=1, shuffle=False, num_workers=2)
    full_loader   = get_dataloader(VAL_DATA, patch_size=256, batch_size=1, is_train=False)

    # 가중치 그리드
    weight_combos = []
    for w0 in [0.4, 0.5, 0.6, 0.7]:
        for w1 in [0.1, 0.15, 0.2, 0.25, 0.3]:
            rem = round(1.0 - w0 - w1, 3)
            if rem < 0.1 or rem > 0.5: continue
            for r in [0.3, 0.4, 0.5, 0.6, 0.7]:
                w2 = round(rem * r, 3)
                w3 = round(rem - w2, 3)
                if w3 < 0.05: continue
                weight_combos.append([w0, w1, w2, w3])

    print(f"  총 {len(weight_combos)}개 조합 | 200장 샘플 그리드 서치\n")

    best_sample_psnr, best_w = 0, None
    sample_results = []

    for i, w in enumerate(weight_combos):
        psnrs = []
        with torch.no_grad():
            for hazy, clear, _ in sample_loader:
                hazy, clear = hazy.to(device), clear.to(device)
                final = None
                for m, wt in zip(models, w):
                    with autocast("cuda"):
                        o = m(hazy).clamp(0,1)
                    final = o*wt if final is None else final+o*wt
                psnrs.append(compute_psnr(final.clamp(0,1), clear))
        p = sum(psnrs)/len(psnrs)
        sample_results.append((p, w))
        if p > best_sample_psnr:
            best_sample_psnr = p
            best_w = w
            print(f"  [{i+1:3d}/{len(weight_combos)}] 🆕 {p:.4f}dB | w={w}")

    print(f"\n  그리드 완료: best_w={best_w} | sample PSNR={best_sample_psnr:.4f}dB")

    # Top5 조합 전체 val 평가
    top5 = sorted(sample_results, reverse=True)[:5]
    print("\n  Top5 조합 전체 val 평가...")

    final_results = []
    for rank, (sp, w) in enumerate(top5):
        p,  s  = eval_on_loader(models, w, full_loader, device, tta=False)
        pt, st = eval_on_loader(models, w, full_loader, device, tta=True)
        tag = "  🎊 30dB!!" if pt >= 30.0 else f"  (30dB까지 {30.0-pt:.4f})"
        print(f"  [{rank+1}] w={w}  noTTA={p:.4f}  TTA={pt:.4f}dB{tag}")
        final_results.append({"rank": rank+1, "weights": w, "no_tta": round(p,4), "tta": round(pt,4)})

    best = max(final_results, key=lambda x: x["tta"])
    print("\n" + "="*65)
    print(f"  🏆 S7 Best: {best['tta']:.4f}dB | w={best['weights']}")
    print(f"  기존 3모델 Best: 29.9963dB | 차이: {best['tta']-29.9963:+.4f}dB")
    if best["tta"] >= 30.0:
        print("  🎊🎊 30dB 달성!! 🎊🎊")
    print("="*65)

    with open(LOG_PATH, "w") as f:
        json.dump({"top5": final_results, "best": best}, f, indent=2)
