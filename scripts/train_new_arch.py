"""
신규 아키텍처 학습 스크립트 — AFDA / FSCA
기존 GB_0149 체크포인트에서 파인튜닝
"""
import os, sys, time, json, argparse, torch, torch.nn as nn, torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.dataset import get_dataloader
from utils.metrics import compute_psnr
from utils.losses import DehazeFormerLoss

DATA_TRAIN = '/home/user/data/RESIDE/RESIDE-6K/train'
DATA_VAL   = '/home/user/data/RESIDE/RESIDE-6K/test'
BASE_CKPT  = '/home/user/experiments/S5_charbonnier_cyclic/best.pth'

class EarlyStopping:
    def __init__(self, patience=15):
        self.patience = patience; self.counter = 0; self.best = None
    def __call__(self, val):
        if self.best is None or val > self.best + 0.005:
            self.best = val; self.counter = 0; return False
        self.counter += 1; return self.counter >= self.patience

def train(arch, size='B', epochs=60, lr=2e-4, batch=8, patch=256,
          out_dir=None, load_partial=True):
    device = torch.device('cuda')
    os.makedirs(out_dir, exist_ok=True)

    # 모델 로드
    if arch == 'afda':
        from models.dehazeformer_afda import build_model_afda
        model = build_model_afda(size).to(device)
        arch_name = 'DehazeFormer-B+AFDA'
    elif arch == 'fsca':
        from models.dehazeformer_fsca import build_model_fsca
        model = build_model_fsca(size).to(device)
        arch_name = 'DehazeFormer-B+FSCA'
    else:
        raise ValueError(f'Unknown arch: {arch}')

    n_params = sum(p.numel() for p in model.parameters())/1e6
    print(f'[{arch_name}] {n_params:.2f}M params')

    # 기존 S5 체크포인트에서 가중치 로드 (공통 레이어만)
    if load_partial and os.path.exists(BASE_CKPT):
        ckpt = torch.load(BASE_CKPT, map_location=device)
        state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
        model_state = model.state_dict()
        loaded, skipped = [], []
        for k, v in state.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v; loaded.append(k)
            else:
                skipped.append(k)
        model.load_state_dict(model_state)
        print(f'[체크포인트] 로드: {len(loaded)}개 / 스킵(신규 모듈): {len(skipped)}개')

    train_loader = get_dataloader(DATA_TRAIN, patch_size=patch, batch_size=batch, is_train=True)
    val_loader   = get_dataloader(DATA_VAL,   patch_size=patch, batch_size=1,     is_train=False)

    criterion = DehazeFormerLoss(l1_w=0.7, perceptual_w=0.3, ssim_w=0.0, freq_w=0.0).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1)
    scaler = GradScaler()
    stopper = EarlyStopping(patience=15)

    best_psnr = 0.0
    log = []

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for hazy, clear, _ in train_loader:
            hazy, clear = hazy.to(device), clear.to(device)
            optimizer.zero_grad()
            with autocast():
                pred = model(hazy)
                loss = criterion(pred, clear)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            total_loss += loss.item()
        scheduler.step()

        # 검증
        if epoch % 5 == 0 or epoch <= 5:
            model.eval()
            psnrs = []
            with torch.no_grad():
                for hazy, clear, _ in val_loader:
                    hazy, clear = hazy.to(device), clear.to(device)
                    pred = model(hazy).clamp(0,1)
                    psnrs.append(float(compute_psnr(pred, clear)))
            val_psnr = sum(psnrs)/len(psnrs)
            avg_loss = total_loss/len(train_loader)
            print(f'[{arch.upper()}] Ep{epoch:03d} | loss={avg_loss:.4f} | val_psnr={val_psnr:.4f}dB | lr={scheduler.get_last_lr()[0]:.2e}')

            log.append({'epoch': epoch, 'loss': avg_loss, 'psnr': val_psnr})
            with open(f'{out_dir}/log.json', 'w') as f:
                json.dump(log, f, indent=2)

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                            'psnr': best_psnr}, f'{out_dir}/best.pth')
                print(f'  ★ Best 갱신: {best_psnr:.4f}dB')

            if stopper(val_psnr):
                print(f'[Early Stop] 최종 Best: {best_psnr:.4f}dB')
                break

    print(f'[완료] {arch.upper()} Best PSNR: {best_psnr:.4f}dB')
    return best_psnr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', required=True, choices=['afda','fsca'])
    parser.add_argument('--size', default='B')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    out_dir = args.out or f'/home/user/experiments/arch_{args.arch}_{args.size}'
    train(args.arch, args.size, args.epochs, args.lr, args.batch, out_dir=out_dir)
