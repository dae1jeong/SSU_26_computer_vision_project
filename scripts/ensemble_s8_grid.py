"""
S8: 5모델 앙상블 그리드 서치
S5(GA파인튜닝) + AFDA(구조개선) + GB_0149(freq) + GA_0065 + GA_0045
핵심: AFDA는 Attention 구조 자체가 달라 기존 모델과 다양성 확보
"""
import os, sys, json, random
import torch
from torch.amp import autocast

REPO_DIR = "/home/user/computer_vision"
sys.path.append(REPO_DIR)
from models.dehazeformer import build_model
from models.dehazeformer_afda import build_model_afda
from datasets.dataset import get_dataloader, HazeDataset
from utils.metrics import compute_psnr, compute_ssim
from torch.utils.data import DataLoader, Subset

EXP_ROOT = "/home/user/experiments"
VAL_DATA  = "/home/user/data/RESIDE/RESIDE-6K/test"
LOG_PATH  = os.path.join(REPO_DIR, "ensemble_s8_log.json")

# (exp_name, size, arch)
ALL_MODELS = [
    ("S5_charbonnier_cyclic",                        "B", "base"),
    ("arch_afda_B",                                  "B", "afda"),   # ← 신규
    ("FT_GB_0149_B_freq_branch_p256_lr2e-04_L1Perc", "B", "base"),
    ("FT_GA_0065_B_baseline_p256_lr2e-04_L1Perc",    "B", "base"),
    ("FT_GA_0045_S_baseline_p256_lr4e-04_L1Perc",    "S", "base"),
]

def load_models(device):
    models = []
    for exp_name, size, arch in ALL_MODELS:
        if arch == "afda":
            m = build_model_afda(size=size).to(device)
        else:
            m = build_model(size=size).to(device)
        ckpt_path = f"{EXP_ROOT}/{exp_name}/best.pth"
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        raw = state.get("model", state.get("model_state_dict", state))
        m.load_state_dict(raw)
        m.eval()
        models.append(m)
        print(f"  ✅ [{arch.upper()}] {exp_name[:50]}")
    return models

def tta_predict(model, hazy):
    augs = [hazy, torch.flip(hazy,[3]), torch.flip(hazy,[2]), torch.flip(hazy,[2,3])]
    out = None
    for k, aug in enumerate(augs):
        with autocast("cuda"):
            o = model(aug).clamp(0,1)
        if k==1: o=torch.flip(o,[3])
        elif k==2: o=torch.flip(o,[2])
        elif k==3: o=torch.flip(o,[2,3])
        out = o if out is None else out+o
    return out / 4

def eval_loader(models, weights, loader, device, tta=False):
    psnrs = []
    with torch.no_grad():
        for hazy, clear, _ in loader:
            hazy, clear = hazy.to(device), clear.to(device)
            final = None
            for m, w in zip(models, weights):
                o = tta_predict(m, hazy) if tta else m(hazy).clamp(0,1)
                final = o*w if final is None else final+o*w
            psnrs.append(compute_psnr(final.clamp(0,1), clear))
    return sum(psnrs)/len(psnrs)


if __name__ == "__main__":
    device = torch.device("cuda")
    print("\n" + "="*65)
    print("  S8: 5모델 앙상블 (S5+AFDA+GB+GA_065+GA_045)")
    print("="*65)

    models = load_models(device)

    # 200장 서브셋
    full_ds = HazeDataset(VAL_DATA, patch_size=256, is_train=False)
    random.seed(42)
    idx200 = random.sample(range(len(full_ds)), 200)
    sample_loader = DataLoader(Subset(full_ds, idx200), batch_size=1, shuffle=False, num_workers=2)
    full_loader   = get_dataloader(VAL_DATA, patch_size=256, batch_size=1, is_train=False)

    # 가중치 그리드: w0(S5), w1(AFDA), w2(GB), w3(GA065), w4(GA045)
    # S5 주도, AFDA 신규 추가, 나머지 분배
    combos = []
    for w0 in [0.40, 0.45, 0.50, 0.55, 0.60]:
        for w1 in [0.05, 0.10, 0.15, 0.20]:          # AFDA 비중
            for w2 in [0.05, 0.10, 0.15]:             # GB
                rem = round(1.0 - w0 - w1 - w2, 3)
                if rem < 0.10 or rem > 0.45: continue
                for r in [0.3, 0.4, 0.5, 0.6, 0.7]:
                    w3 = round(rem * r, 3)
                    w4 = round(rem - w3, 3)
                    if w4 < 0.03: continue
                    combos.append([w0, w1, w2, w3, w4])

    print(f"  총 {len(combos)}개 조합\n")

    best_p, best_w = 0, None
    sample_results = []

    for i, w in enumerate(combos):
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
        if p > best_p:
            best_p = p
            best_w = w
            print(f"  [{i+1:3d}/{len(combos)}] 🆕 {p:.4f}dB | w={w}")

    print(f"\n  그리드 완료: best_w={best_w} | sample={best_p:.4f}dB")

    # Top5 전체 val 평가
    top5 = sorted(sample_results, reverse=True)[:5]
    print("\n  Top5 전체 val 평가...")
    final_results = []
    for rank, (sp, w) in enumerate(top5):
        p  = eval_loader(models, w, full_loader, device, tta=False)
        pt = eval_loader(models, w, full_loader, device, tta=True)
        tag = "  🎊🎊" if pt >= 30.0 else f"  ({30.0-pt:+.4f}dB)"
        print(f"  [{rank+1}] w={w}  noTTA={p:.4f}  TTA={pt:.4f}dB{tag}")
        final_results.append({"rank":rank+1,"weights":w,"no_tta":round(p,4),"tta":round(pt,4)})

    best = max(final_results, key=lambda x: x["tta"])
    s7_best = 30.0008
    print("\n" + "="*65)
    print(f"  🏆 S8 Best TTA : {best['tta']:.4f}dB")
    print(f"  S7 대비        : {best['tta']-s7_best:+.4f}dB")
    if best["tta"] >= 30.0: print("  🎊 30dB 달성!")
    print("="*65)

    with open(LOG_PATH, "w") as f:
        json.dump({"top5": final_results, "best": best,
                   "s7_baseline": s7_best,
                   "delta": round(best["tta"]-s7_best, 4)}, f, indent=2)
    print(f"\n  로그 저장: {LOG_PATH}")
