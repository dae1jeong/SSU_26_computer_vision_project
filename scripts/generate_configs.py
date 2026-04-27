"""
실험 Grid 자동 생성 스크립트
모든 조합의 config.json을 configs/ 폴더에 생성
"""

import os
import json
import itertools

# ───────────────────────────────────────────────
# 실험 변수 정의
# ───────────────────────────────────────────────

BASE = {
    "train_data":    "/data/RESIDE/ITS",
    "val_data":      "/data/RESIDE/SOTS/indoor",
    "exp_root":      "/home/user/experiments",
    "epochs":        200,
    "val_freq":      10,
    "weight_decay":  0.02,
}

GRID = {
    "model_size":       ["S", "B"],          # 모델 사이즈
    "patch_size":       [128, 256],           # 입력 패치 크기
    "batch_size":       [8],                  # GPU 메모리에 맞게
    "lr":               [1e-4, 2e-4, 4e-4],  # 학습률
    "loss_l1":          [1.0],
    "loss_perceptual":  [0.0, 0.1],           # Perceptual Loss 유무
    "loss_ssim":        [0.0, 0.2],           # SSIM Loss 유무
    "loss_freq":        [0.0, 0.05],          # Frequency Loss 유무
}

# ───────────────────────────────────────────────
# 조합 생성
# ───────────────────────────────────────────────

keys   = list(GRID.keys())
values = list(GRID.values())
combos = list(itertools.product(*values))

os.makedirs("configs", exist_ok=True)

for i, combo in enumerate(combos):
    cfg = dict(BASE)
    for k, v in zip(keys, combo):
        cfg[k] = v

    # 실험 이름 자동 생성
    exp_name = (
        f"exp{i:04d}"
        f"_dehaze{cfg['model_size']}"
        f"_p{cfg['patch_size']}"
        f"_lr{cfg['lr']:.0e}"
        f"_perc{cfg['loss_perceptual']}"
        f"_ssim{cfg['loss_ssim']}"
        f"_freq{cfg['loss_freq']}"
    )
    cfg['exp_name'] = exp_name

    fname = os.path.join("configs", f"{exp_name}.json")
    with open(fname, 'w') as f:
        json.dump(cfg, f, indent=2)

print(f"✅ {len(combos)}개 실험 config 생성 완료!")
print(f"   configs/ 폴더 확인")

# 실험 목록 CSV 저장
import csv
with open("configs/experiment_list.csv", 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['exp_name'] + keys)
    writer.writeheader()
    for i, combo in enumerate(combos):
        row = {'exp_name': f"exp{i:04d}"}
        for k, v in zip(keys, combo):
            row[k] = v
        writer.writerow(row)

print(f"   experiment_list.csv 생성 완료")
