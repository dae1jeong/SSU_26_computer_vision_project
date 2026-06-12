"""
S6: S5 best 기반 추가 Cyclic Fine-tuning
S5_charbonnier_cyclic best.pth → lr=5e-6, T_0=20, 3사이클 60ep
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

BASE_EXP   = 'S5_charbonnier_cyclic'
EXP_NAME   = 'S6_s5_cyclic_extra'
EXP_ROOT   = '/home/user/experiments'
TRAIN_DATA = '/home/user/data/RESIDE/RESIDE-6K/train'
VAL_DATA   = '/home/user/data/RESIDE/RESIDE-6K/test'
REPO_DIR   = '/home/user/computer_vision'

PATCH_SIZE  = 256
BATCH_SIZE  = 4
LR_INIT     = 5e-6
T_0         = 20
N_CYCLES    = 3
TOTAL_EP    = T_0 * N_CYCLES   # 60
LR_MIN      = 1e-7
EPS_CHARB   = 1e-3
STATE_PSNR  = 29.9096          # S5 best PSNR 하드코딩

os.makedirs(f'{EXP_ROOT}/{EXP_NAME}', exist_ok=True)
LOG_PATH  = f'{REPO_DIR}/finetune_scenario_log.json'
PROG_PATH = f'{EXP_ROOT}/{EXP_NAME}/progress.json'
BEST_PATH = f'{EXP_ROOT}/{EXP_NAME}/best.pth'


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


def save_progress(ep, cur_psnr, best_psnr, best_ep):
    with open(PROG_PATH, 'w') as f:
        json.dump({'current_epoch': ep, 'current_psnr': cur_psnr,
                   'best_psnr': best_psnr, 'best_epoch': best_ep}, f)


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


if __name__ == '__main__':
    device = torch.device('cuda')

    print('\n' + '='*65)
    print(f'  🔥 [S6] S5 기반 추가 Cyclic Fine-tuning')
    print(f'  기반: {BASE_EXP} ({STATE_PSNR}dB)')
    print(f'  lr: {LR_INIT} | T_0={T_0} | {N_CYCLES}사이클 = {TOTAL_EP}ep')
    print(f'  Loss: Charbonnier (eps={EPS_CHARB})')
    print('='*65)

    model = build_model(size='B').to(device)
    ckpt  = f'{EXP_ROOT}/{BASE_EXP}/best.pth'
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state['model'] if 'model' in state else state)
    print(f'\n  ✅ 체크포인트 로드: {ckpt}')

    train_loader = get_dataloader(TRAIN_DATA, patch_size=PATCH_SIZE,
                                  batch_size=BATCH_SIZE, is_train=True)
    val_loader   = get_dataloader(VAL_DATA,   patch_size=PATCH_SIZE,
                                  batch_size=1, is_train=False)

    criterion = CharbonnierLoss(eps=EPS_CHARB)
    optimizer = AdamW(model.parameters(), lr=LR_INIT, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=1, eta_min=LR_MIN)
    scaler    = GradScaler()

    best_psnr = STATE_PSNR
    best_ep   = 0

    print(f'\n  기준 PSNR: {STATE_PSNR}dB')
    print(f'  총 {TOTAL_EP} epoch 시작\n')

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

        if ep % 5 == 0 or ep == TOTAL_EP:
            psnr, ssim = evaluate(model, val_loader, device)
            improved = ''
            if psnr > best_psnr:
                best_psnr = psnr
                best_ep   = ep
                torch.save({'model': model.state_dict(), 'psnr': psnr, 'epoch': ep}, BEST_PATH)
                improved = '  🆕 Best!'
                if psnr >= 30.0:
                    improved += '  🎊 30dB 달성!!'
            save_progress(ep, round(psnr, 4), round(best_psnr, 4), best_ep)
            cycle = (ep - 1) // T_0 + 1
            print(f'  ep{ep:03d}/cycle{cycle} | loss={avg_loss:.4f} | lr={cur_lr:.2e} | '
                  f'PSNR={psnr:.4f}dB | Best={best_psnr:.4f}dB{improved}')
        else:
            save_progress(ep, 0, round(best_psnr, 4), best_ep)

    # 로그 저장
    try:
        with open(LOG_PATH) as f:
            log = json.load(f)
    except:
        log = {'scenarios': [], 'global_best': {'psnr': 0, 'exp': ''}}

    log['scenarios'].append({
        'scenario_id': 'S6',
        'name': EXP_NAME,
        'base_psnr': STATE_PSNR,
        'result_psnr': round(best_psnr, 4),
        'improvement': round(best_psnr - STATE_PSNR, 4),
        'epochs_run': TOTAL_EP,
        'status': 'done',
        'notes': f'S5 기반 Charbonnier Cyclic lr={LR_INIT}, T_0={T_0}, {N_CYCLES}사이클',
        'reached_30db': best_psnr >= 30.0,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    })
    if best_psnr > log.get('global_best', {}).get('psnr', 0):
        log['global_best'] = {'psnr': round(best_psnr, 4), 'exp': EXP_NAME,
                               'updated': time.strftime('%Y-%m-%d %H:%M:%S')}
    with open(LOG_PATH, 'w') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    print(f'\n  ✅ S6 완료: {STATE_PSNR}dB → {best_psnr:.4f}dB (best ep{best_ep})')
    print(f'  30dB {"달성!! 🎊" if best_psnr >= 30.0 else f"까지 {30.0-best_psnr:.4f}dB 남음"}')
