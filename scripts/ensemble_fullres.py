"""
Full-Resolution 앙상블 + TTA 평가
- 3모델 한 번에 로드, 이미지 하나씩 원본 해상도로 추론
- 32-align padding
"""
import os, sys, json, time
import torch
import torch.nn.functional as F
from torch.amp import autocast
from PIL import Image
import torchvision.transforms.functional as TF

REPO_DIR = "/home/user/computer_vision"
sys.path.append(REPO_DIR)
from models.dehazeformer import build_model
from utils.metrics import compute_psnr, compute_ssim

EXP_ROOT = "/home/user/experiments"
VAL_DATA = "/home/user/data/RESIDE/RESIDE-6K/test"
LOG_PATH = os.path.join(REPO_DIR, "ensemble_fullres_log.json")

ENSEMBLE_CFG = [
    ("S5_charbonnier_cyclic",                       "B", 0.5),
    ("FT_GA_0065_B_baseline_p256_lr2e-04_L1Perc",  "B", 0.3),
    ("FT_GA_0045_S_baseline_p256_lr4e-04_L1Perc",  "S", 0.2),
]

def load_models(device):
    models, weights = [], []
    for exp_name, size, w in ENSEMBLE_CFG:
        m = build_model(size=size).to(device)
        state = torch.load(f"{EXP_ROOT}/{exp_name}/best.pth", map_location=device, weights_only=False)
        m.load_state_dict(state["model"] if "model" in state else state)
        m.eval()
        models.append(m)
        weights.append(w)
        print(f"  ✅ {exp_name[:45]} (w={w})")
    return models, weights

def infer(model, x, align=32):
    B, C, H, W = x.shape
    pad_h = (align - H % align) % align
    pad_w = (align - W % align) % align
    xp = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    with torch.no_grad():
        with autocast("cuda"):
            out = model(xp).clamp(0, 1)
    return out[:, :, :H, :W]

def get_items(data_dir):
    hazy_dir = os.path.join(data_dir, "hazy")
    gt_dir   = os.path.join(data_dir, "GT") if os.path.exists(os.path.join(data_dir, "GT")) else os.path.join(data_dir, "clear")
    files    = sorted([f for f in os.listdir(hazy_dir) if f.endswith((".png",".jpg"))])
    items = []
    for fname in files:
        stem, ext = os.path.splitext(fname)
        cn = fname
        if not os.path.exists(os.path.join(gt_dir, cn)):
            for e in [ext, '.png', '.jpg']:
                cand = stem.split('_')[0] + e
                if os.path.exists(os.path.join(gt_dir, cand)):
                    cn = cand; break
        cp = os.path.join(gt_dir, cn)
        if os.path.exists(cp):
            items.append((os.path.join(hazy_dir, fname), cp))
    return items

def evaluate(models, weights, items, device, tta=False):
    psnrs, ssims = [], []
    for idx, (hp, cp) in enumerate(items):
        hazy  = TF.to_tensor(Image.open(hp).convert("RGB")).unsqueeze(0).to(device)
        clear = TF.to_tensor(Image.open(cp).convert("RGB")).unsqueeze(0).to(device)

        if tta:
            augs = [hazy, torch.flip(hazy,[3]), torch.flip(hazy,[2]), torch.flip(hazy,[2,3])]
            final = None
            for m, w in zip(models, weights):
                s = None
                for k, aug in enumerate(augs):
                    o = infer(m, aug)
                    if k==1: o=torch.flip(o,[3])
                    elif k==2: o=torch.flip(o,[2])
                    elif k==3: o=torch.flip(o,[2,3])
                    s = o if s is None else s+o
                final = (s/4)*w if final is None else final+(s/4)*w
        else:
            final = None
            for m, w in zip(models, weights):
                o = infer(m, hazy)
                final = o*w if final is None else final+o*w

        psnrs.append(compute_psnr(final.clamp(0,1), clear))
        ssims.append(compute_ssim(final.clamp(0,1), clear))

        if (idx+1) % 100 == 0:
            print(f"    [{idx+1}/{len(items)}] avg PSNR: {sum(psnrs)/len(psnrs):.4f}dB")

    return sum(psnrs)/len(psnrs), sum(ssims)/len(ssims)


if __name__ == "__main__":
    device = torch.device("cuda")
    print("\n" + "="*65)
    print("  🔬 Full-Resolution 앙상블 + TTA")
    print("  3모델 동시 로드 | 원본 해상도 | 32-align padding")
    print("="*65)

    models, weights = load_models(device)
    items = get_items(VAL_DATA)
    print(f"  Val: {len(items)}개 | VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    log = {"results": [], "best": {"psnr": 0, "method": ""}}

    stages = [("fullres_no_tta", False), ("fullres_tta", True)]
    labels = ["Full-Res 앙상블 (no TTA)", "Full-Res 앙상블 + TTA 4방향"]

    for (stage, use_tta), label in zip(stages, labels):
        print(f"\n  [{label}] 평가 중...")
        t0 = time.time()
        p, s = evaluate(models, weights, items, device, tta=use_tta)
        elapsed = int(time.time()-t0)
        star = "  🎊 30dB 달성!!" if p >= 30.0 else f"  (30dB까지 {30.0-p:.4f}dB)"
        print(f"  ✅ {label}: PSNR={p:.4f}dB | SSIM={s:.4f} | {elapsed}s{star}")
        log["results"].append({"stage": stage, "psnr": round(p,4), "ssim": round(s,4), "elapsed": elapsed})
        if p > log["best"]["psnr"]:
            log["best"] = {"psnr": round(p,4), "method": stage}

    best = log["best"]["psnr"]
    print("\n" + "="*65)
    print(f"  🏆 Best: {log['best']['method']} → {best:.4f}dB")
    print(f"  기존 (256crop+TTA): 29.9963dB | 차이: {best-29.9963:+.4f}dB")
    if best >= 30.0:
        print("  🎊🎊 30dB 달성!! 🎊🎊")
    print("="*65)

    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"  로그 저장: {LOG_PATH}")
