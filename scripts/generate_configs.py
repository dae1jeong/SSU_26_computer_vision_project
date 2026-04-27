"""
GPU 서버용 실험 config 자동 생성 (RESIDE-6K 기준)
"""

import os
import json
import itertools
import csv

BASE = {
    "train_data":    "/home/user/data/RESIDE/train",
    "val_data":      "/home/user/data/RESIDE/test",
    "exp_root":      "/home/user/experiments",
    "epochs":        100,
    "val_freq":      5,
    "weight_decay":  0.02,
    "early_stop_patience": 15,
}

# Group A: Baseline
GROUP_A = list(itertools.product(
    ["T", "S", "B", "L"],   # model_size
    ["baseline"],            # architecture
    [128, 256],              # patch_size (L은 128만)
    [1e-4, 2e-4, 4e-4],     # lr
    ["L1", "L1+Perc", "L1+SSIM", "L1+Perc+SSIM+Freq"],  # loss
))

# Group B: Frequency Branch
GROUP_B = list(itertools.product(
    ["T", "S", "B"], ["freq_branch"],
    [128, 256], [1e-4, 2e-4, 4e-4],
    ["L1", "L1+Perc", "L1+SSIM", "L1+Perc+SSIM+Freq"],
))

# Group C: Deformable Attention
GROUP_C = list(itertools.product(
    ["T", "S", "B"], ["deformable_attn"],
    [128, 256], [1e-4, 2e-4, 4e-4],
    ["L1", "L1+Perc", "L1+SSIM", "L1+Perc+SSIM+Freq"],
))

# Group D: Channel Attention
GROUP_D = list(itertools.product(
    ["S", "B"], ["channel_attn"],
    [128, 256], [1e-4, 2e-4, 4e-4],
    ["L1", "L1+Perc+SSIM+Freq"],
))

# Group E: Freq + Deformable 결합
GROUP_E = list(itertools.product(
    ["S", "B"], ["freq_deformable"],
    [128, 256], [1e-4, 2e-4, 4e-4],
    ["L1", "L1+Perc+SSIM+Freq"],
))

LOSS_MAP = {
    "L1":               {"loss_l1": 1.0, "loss_perceptual": 0.0, "loss_ssim": 0.0, "loss_freq": 0.0},
    "L1+Perc":          {"loss_l1": 1.0, "loss_perceptual": 0.1, "loss_ssim": 0.0, "loss_freq": 0.0},
    "L1+SSIM":          {"loss_l1": 1.0, "loss_perceptual": 0.0, "loss_ssim": 0.2, "loss_freq": 0.0},
    "L1+Perc+SSIM+Freq":{"loss_l1": 1.0, "loss_perceptual": 0.1, "loss_ssim": 0.2, "loss_freq": 0.05},
}

GROUP_LABELS = [
    ("A", GROUP_A),
    ("B", GROUP_B),
    ("C", GROUP_C),
    ("D", GROUP_D),
    ("E", GROUP_E),
]

os.makedirs("configs", exist_ok=True)
all_rows = []
total = 0

for group_label, group in GROUP_LABELS:
    for i, (model, arch, patch, lr, loss_name) in enumerate(group):
        # L 모델은 patch=128만
        if model == "L" and patch == 256:
            continue

        cfg = dict(BASE)
        cfg["model_size"]   = model
        cfg["architecture"] = arch
        cfg["patch_size"]   = patch
        cfg["lr"]           = lr
        cfg["batch_size"]   = 4 if patch == 256 else 8
        cfg.update(LOSS_MAP[loss_name])
        cfg["loss_name"]    = loss_name

        exp_name = f"G{group_label}_{total:04d}_{model}_{arch}_p{patch}_lr{lr:.0e}_{loss_name.replace('+','')}"
        cfg["exp_name"] = exp_name

        fname = os.path.join("configs", f"{exp_name}.json")
        with open(fname, "w") as f:
            json.dump(cfg, f, indent=2)

        all_rows.append({
            "group": group_label, "exp_name": exp_name,
            "model": model, "arch": arch, "patch": patch,
            "lr": lr, "loss": loss_name
        })
        total += 1

# CSV 저장
with open("configs/experiment_list.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["group","exp_name","model","arch","patch","lr","loss"])
    writer.writeheader()
    writer.writerows(all_rows)

print(f"✅ 총 {total}개 실험 config 생성 완료!")
for g, grp in GROUP_LABELS:
    cnt = sum(1 for r in all_rows if r["group"] == g)
    print(f"   Group {g}: {cnt}개")
