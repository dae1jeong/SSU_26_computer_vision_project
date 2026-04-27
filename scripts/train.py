"""
학습 스크립트
"""

import os
import sys
import time
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dehazeformer import build_model
from datasets.dataset import get_dataloader
from utils.metrics import compute_psnr, compute_ssim
from utils.losses import DehazeFormerLoss


def train(config):
    # ─── 디바이스 ───
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    # ─── 모델 ───
    model = build_model(size=config['model_size']).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] DehazeFormer-{config['model_size']} | Params: {n_params/1e6:.2f}M")

    # ─── 데이터 ───
    train_loader = get_dataloader(
        config['train_data'], patch_size=config['patch_size'],
        batch_size=config['batch_size'], is_train=True
    )
    val_loader = get_dataloader(
        config['val_data'], patch_size=config['patch_size'],
        batch_size=1, is_train=False
    )

    # ─── Loss / Optimizer / Scheduler ───
    criterion = DehazeFormerLoss(
        l1_w=config['loss_l1'],
        perceptual_w=config['loss_perceptual'],
        ssim_w=config['loss_ssim'],
        freq_w=config['loss_freq']
    ).to(device)

    optimizer = optim.AdamW(model.parameters(),
                            lr=config['lr'],
                            weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    scaler    = GradScaler()

    # ─── 로깅 ───
    exp_dir = os.path.join(config['exp_root'], config['exp_name'])
    os.makedirs(exp_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(exp_dir, 'logs'))
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    best_psnr = 0.0
    results = []

    for epoch in range(1, config['epochs'] + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for hazy, clear, _ in train_loader:
            hazy, clear = hazy.to(device), clear.to(device)
            optimizer.zero_grad()
            with autocast():
                pred = model(hazy)
                loss = criterion(pred, clear)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)
        elapsed = time.time() - t0

        # ── Validate ──
        if epoch % config['val_freq'] == 0:
            model.eval()
            psnrs, ssims = [], []
            with torch.no_grad():
                for hazy, clear, _ in val_loader:
                    hazy, clear = hazy.to(device), clear.to(device)
                    pred = model(hazy).clamp(0, 1)
                    psnrs.append(compute_psnr(pred, clear))
                    ssims.append(compute_ssim(pred, clear))

            avg_psnr = sum(psnrs) / len(psnrs)
            avg_ssim = sum(ssims) / len(ssims)

            print(f"[{epoch:03d}/{config['epochs']}] "
                  f"Loss={train_loss:.4f} | PSNR={avg_psnr:.2f}dB | SSIM={avg_ssim:.4f} | {elapsed:.0f}s")

            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('PSNR/val',   avg_psnr,   epoch)
            writer.add_scalar('SSIM/val',   avg_ssim,   epoch)

            row = {'epoch': epoch, 'loss': train_loss,
                   'psnr': avg_psnr, 'ssim': avg_ssim}
            results.append(row)

            # Best 체크포인트 저장
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                torch.save({'epoch': epoch, 'model': model.state_dict(),
                            'psnr': best_psnr, 'config': config},
                           os.path.join(exp_dir, 'best.pth'))
                print(f"  ✅ Best saved! PSNR={best_psnr:.2f}dB")

    # ─── 결과 저장 ───
    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    writer.close()
    print(f"\n[Done] {config['exp_name']} | Best PSNR={best_psnr:.2f}dB")
    return best_psnr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    train(config)
