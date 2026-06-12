"""
2단계: Charbonnier Loss + Cyclic Fine-tuning
S3 best.pth 기반, lr 1e-5 → 3e-6 → 1e-6 Cosine 반복
Charbonnier Loss: sqrt((pred-target)^2 + eps^2) — L1보다 PSNR 직접 최적화에 유리
"""

import os, sys, json, time, math
sys.path.append('/home/user/computer_vision')

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from models.dehazeformer import build_model
from datasets.dataset import get_dataloader
from utils.metrics import compute_psnr, compute_ssim

# ── 설정 ─────────────────────────────────────────────────────
BASE_EXP   = "S3_best_extend_lr5e6"
EXP_NAME   = "S5_charbonnier_cyclic"
EXP_ROOT   = "/home/user/experiments"
TRAIN_DATA = "/home/user/data/RESIDE/RESIDE-6K/train"
VAL_DATA   = "/home/user/data/RESIDE/RESIDE-6K/test"
REPO_DIR   = "/home/user/computer_vision"

PATCH_SIZE = 256
BATCH_SIZE = 4
LR_INIT    = 1e-5      # S3 완료 lr 수준
T_0        = 30        # Cosine 주기
N_CYCLES   = 3         # 3번 반복 (30+30+30 = 90 epoch)
TOTAL_EP   = T_0 * N_CYCLES
LR_MIN     = 1e-7
EPS_CHARB  = 1e-3      # Charbonnier epsilon

os.makedirs(f"{EXP_ROOT}/{EXP_NAME}", exist_ok=True)
LOG_PATH  = f"{REPO_DIR}/finetune_scenario_log.json"
PROG_PATH = f"{EXP_ROOT}/{EXP_NAME}/progress.json"


# ── Charbonnier Loss ──────────────────────────────────────────
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


def save_progress(ep, cur_psnr, best_psnr, best_ep):
    with open(PROG_PATH, 'w') as f:
        json.dump({"current_epoch": ep, "current_psnr": cur_psnr,
                   "best_psnr": best_psnr, "best_epoch": best_ep}, f)


def evaluate(model, val_loader, device):
    model.eval()
    psnrs, ssims = [], []
    with torch.no_grad():
        for hazy, clear, _ in val_loader:
            hazy, clear = hazy.to(device), clear.to(device)
            with autocast('cuda'):
                pred = model(hazy).clamp(0, 1)
            psnrs.append(compute_psnr(pred, clear))
            ssims.append(compute_ssim(pred, clear))
    return sum(psnrs)/len(psnrs), sum(ssims)/len(ssims)


if __name__ == "__main__":
    device = torch.device('cuda')

    print("\n" + "="*65)
    print(f"  🔥 [S5] Charbonnier + Cyclic Fine-tuning")
    print(f"  기반: {BASE_EXP} (29.7915dB)")
    print(f"  lr: {LR_INIT} | T_0={T_0} | {N_CYCLES}사이클 = {TOTAL_EP}ep")
    print(f"  Loss: Charbonnier (eps={EPS_CHARB})")
    print("="*65)

    # 모델 로드
    model = build_model(size='B').to(device)
    ckpt = f"{EXP_ROOT}/{BASE_EXP}/best.pth"
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state['model'] if 'model' in state else state)
    print(f"\n  ✅ 체크포인트 로드: {ckpt}")

    # 데이터
    train_loader = get_dataloader(TRAIN_DATA, patch_size=PATCH_SIZE,
                                   batch_size=BATCH_SIZE, is_train=True)
    val_loader   = get_dataloader(VAL_DATA, patch_size=PATCH_SIZE,
                                   batch_size=1, is_train=False)

    # 옵티마이저 & 스케줄러
    criterion = CharbonnierLoss(eps=EPS_CHARB)
    optimizer = AdamW(model.parameters(), lr=LR_INIT, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=1, eta_min=LR_MIN)
    scaler    = GradScaler()

    # 기준 PSNR
    state_psnr = 29.7915
    best_psnr  = state_psnr
    best_ep    = 0
    best_path  = f"{EXP_ROOT}/{EXP_NAME}/best.pth"

    print(f"\n  기준 PSNR: {state_psnr}dB")
    print(f"  총 {TOTAL_EP} epoch 시작\n")

    for ep in range(1, TOTAL_EP + 1):
        model.train()
        losses = []
        for hazy, clear, _ in train_loader:
            hazy, clear = hazy.to(device), clear.to(device)
            optimizer.zero_grad()
            with autocast('cuda'):
                pred = model(hazy).clamp(0, 1)
                loss = criterion(pred, clear)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        scheduler.step()

        avg_loss = sum(losses) / len(losses)
        cur_lr   = optimizer.param_groups[0]['lr']

        # 5 epoch마다 val 평가
        if ep % 5 == 0 or ep == TOTAL_EP:
            psnr, ssim = evaluate(model, val_loader, device)
            improved = ""
            if psnr > best_psnr:
                best_psnr = psnr
                best_ep   = ep
                torch.save({'model': model.state_dict(), 'psnr': psnr, 'epoch': ep}, best_path)
                improved = "  🆕 Best!"
                if psnr >= 30.0:
                    improved += "  🎊 30dB 달성!!"

            save_progress(ep, round(psnr, 4), round(best_psnr, 4), best_ep)
            cycle = (ep - 1) // T_0 + 1
            print(f"  ep{ep:03d}/cycle{cycle} | loss={avg_loss:.4f} | lr={cur_lr:.2e} | "
                  f"PSNR={psnr:.4f}dB | Best={best_psnr:.4f}dB{improved}")
        else:
            save_progress(ep, 0, round(best_psnr, 4), best_ep)

    # 로그 저장
    with open(LOG_PATH) as f: log = json.load(f)
    log['scenarios'].append({
        'scenario_id': 'S5',
        'name': EXP_NAME,
        'base_psnr': state_psnr,
        'result_psnr': round(best_psnr, 4),
        'improvement': round(best_psnr - state_psnr, 4),
        'epochs_run': TOTAL_EP,
        'status': 'done',
        'notes': f'Charbonnier Loss + Cyclic lr {LR_INIT}→{LR_MIN}, T_0={T_0}, {N_CYCLES}사이클',
        'reached_30db': best_psnr >= 30.0,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    if best_psnr > log['global_best']['psnr']:
        log['global_best'] = {'psnr': round(best_psnr, 4), 'exp': EXP_NAME,
                               'updated': time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(LOG_PATH, 'w') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ S5 완료: {state_psnr}dB → {best_psnr:.4f}dB (best ep{best_ep})")
    print(f"  30dB {'달성!! 🎊' if best_psnr >= 30.0 else f'까지 {30.0-best_psnr:.4f}dB 남음'}")
