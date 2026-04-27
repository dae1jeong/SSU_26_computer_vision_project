"""
Early Stopping 포함 학습 스크립트 (GPU 서버용)
"""

import os
import sys
import time
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


class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.01):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best      = None

    def __call__(self, val_psnr):
        if self.best is None or val_psnr > self.best + self.min_delta:
            self.best   = val_psnr
            self.counter = 0
            return False  # 계속 진행
        self.counter += 1
        return self.counter >= self.patience  # True = 중단


def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

    model = build_model(size=config['model_size']).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] DehazeFormer-{config['model_size']} ({config['architecture']}) | {n_params/1e6:.2f}M params")

    train_loader = get_dataloader(config['train_data'], patch_size=config['patch_size'],
                                  batch_size=config['batch_size'], is_train=True)
    val_loader   = get_dataloader(config['val_data'],   patch_size=config['patch_size'],
                                  batch_size=1, is_train=False)

    criterion = DehazeFormerLoss(
        l1_w=config['loss_l1'], perceptual_w=config['loss_perceptual'],
        ssim_w=config['loss_ssim'], freq_w=config['loss_freq']
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    scaler    = GradScaler()
    stopper   = EarlyStopping(patience=config.get('early_stop_patience', 15))

    exp_dir = os.path.join(config['exp_root'], config['exp_name'])
    os.makedirs(exp_dir, exist_ok=True)
    writer  = SummaryWriter(os.path.join(exp_dir, 'logs'))

    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    best_psnr, results = 0.0, []

    for epoch in range(1, config['epochs'] + 1):
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

        # Validation
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
            elapsed  = time.time() - t0

            print(f"[{epoch:03d}/{config['epochs']}] Loss={train_loss:.4f} | "
                  f"PSNR={avg_psnr:.2f}dB | SSIM={avg_ssim:.4f} | {elapsed:.0f}s")

            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('PSNR/val',   avg_psnr,   epoch)
            writer.add_scalar('SSIM/val',   avg_ssim,   epoch)
            results.append({'epoch': epoch, 'loss': train_loss, 'psnr': avg_psnr, 'ssim': avg_ssim})

            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                torch.save({'epoch': epoch, 'model': model.state_dict(),
                            'psnr': best_psnr, 'config': config},
                           os.path.join(exp_dir, 'best.pth'))
                print(f"  ✅ Best! PSNR={best_psnr:.2f}dB")

            # Early Stopping
            if stopper(avg_psnr):
                print(f"  ⏹  Early stopping at epoch {epoch} (patience={stopper.patience})")
                break

    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump({'best_psnr': best_psnr, 'history': results}, f, indent=2)

    writer.close()
    print(f"[Done] {config['exp_name']} | Best PSNR={best_psnr:.2f}dB")
    return best_psnr


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    train(config)
