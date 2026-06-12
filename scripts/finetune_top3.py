"""
Top3 모델 Warm Restart 파인튜닝 스크립트
- 대상: GB_0149, GA_0065, GA_0045
- 전략: Warm Restart + EMA + 낮은 lr
- 예상 기간: 3~4일
"""

import os, sys, json, time, copy, math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast

REPO_DIR = "/home/user/computer_vision"
sys.path.append(REPO_DIR)

from models.dehazeformer import build_model
from datasets.dataset import get_dataloader
from utils.metrics import compute_psnr, compute_ssim
from utils.losses import DehazeFormerLoss

# ── EMA (Exponential Moving Average) ──────────────────────────
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] +
                    (1.0 - self.decay) * param.data
                )

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


# ── 파인튜닝 실행 함수 ─────────────────────────────────────────
def finetune(exp_name, model_size, architecture, ft_config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 원본 체크포인트 경로
    ckpt_path = os.path.join(ft_config['exp_root'], exp_name, 'best.pth')
    ft_exp_name = f"FT_{exp_name}"
    ft_dir = os.path.join(ft_config['exp_root'], ft_exp_name)
    os.makedirs(ft_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  🔧 파인튜닝 시작: {ft_exp_name}")
    print(f"  체크포인트: {ckpt_path}")
    print(f"  lr: {ft_config['lr_start']} | epochs: {ft_config['epochs']}")
    print(f"  T_0: {ft_config['T_0']} | T_mult: {ft_config['T_mult']} | EMA: {ft_config['use_ema']}")
    print(f"{'='*65}\n")

    # 모델 로드
    model = build_model(size=model_size).to(device)
    state = torch.load(ckpt_path, map_location=device)
    if 'model' in state:
        model.load_state_dict(state['model'])
        prev_psnr = state.get('psnr', 0)
        print(f"  ✅ 체크포인트 로드 완료 (기존 best PSNR: {prev_psnr:.4f}dB)")
    elif 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
        print(f"  ✅ 체크포인트 로드 완료")
    else:
        model.load_state_dict(state)
        print(f"  ✅ 체크포인트 로드 완료")

    # 데이터로더
    train_loader = get_dataloader(
        ft_config['train_data'], patch_size=ft_config['patch_size'],
        batch_size=ft_config['batch_size'], is_train=True
    )
    val_loader = get_dataloader(
        ft_config['val_data'], patch_size=ft_config['patch_size'],
        batch_size=1, is_train=False
    )

    # Loss, Optimizer
    criterion = DehazeFormerLoss(
        l1_w=1.0, perceptual_w=0.1, ssim_w=0.0, freq_w=0.0
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=ft_config['lr_start'],
        weight_decay=1e-4
    )

    # Warm Restart 스케줄러
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=ft_config['T_0'],      # 첫 번째 재시작 주기
        T_mult=ft_config['T_mult'], # 이후 주기 배수
        eta_min=ft_config['lr_min'] # 최소 lr
    )

    scaler = GradScaler('cuda')
    ema = EMA(model, decay=0.999) if ft_config['use_ema'] else None

    # 기존 best PSNR을 초기값으로 설정 → 파인튜닝이 기존보다 나빠지면 저장 안 함
    state_psnr = state.get('psnr', 0.0) if isinstance(state, dict) else 0.0
    best_psnr = state_psnr
    best_epoch = 0
    results = []

    # Restart-aware 얼리스탑: 각 주기 끝에서 체크
    # 주기: T_0=40, T_1=80, T_2=80
    T_0 = ft_config['T_0']
    T_mult = ft_config['T_mult']
    restart_boundaries = []
    t = T_0
    ep = 0
    while ep + t <= ft_config['epochs']:
        ep += t
        restart_boundaries.append(ep)
        t = t * T_mult
    print(f"  Restart 경계 epoch: {restart_boundaries}")

    cycle_best_psnr = 0.0   # 현재 주기 최고 PSNR
    prev_cycle_best = 0.0   # 직전 주기 최고 PSNR
    cycle_improve_thresh = 0.005  # 주기 간 최소 개선폭 (dB)

    for epoch in range(1, ft_config['epochs'] + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for hazy, clear, _ in train_loader:
            hazy, clear = hazy.to(device), clear.to(device)
            optimizer.zero_grad()
            with autocast('cuda'):
                pred = model(hazy)
                loss = criterion(pred, clear)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

            # EMA 업데이트
            if ema:
                ema.update(model)

        scheduler.step()
        train_loss /= len(train_loader)
        elapsed = int(time.time() - t0)
        cur_lr = optimizer.param_groups[0]['lr']

        # NaN 감지
        if math.isnan(train_loss) or math.isinf(train_loss):
            print(f"  ❌ NaN 감지 at epoch {epoch} → 중단")
            break

        # Validation (매 5 epoch)
        if epoch % 5 == 0:
            # EMA 가중치로 평가
            if ema:
                ema.apply_shadow(model)

            model.eval()
            psnrs, ssims = [], []
            with torch.no_grad():
                for hazy, clear, _ in val_loader:
                    hazy, clear = hazy.to(device), clear.to(device)
                    with autocast('cuda'):
                        pred = model(hazy).clamp(0, 1)
                    psnrs.append(compute_psnr(pred, clear))
                    ssims.append(compute_ssim(pred, clear))

            val_psnr = sum(psnrs) / len(psnrs)
            val_ssim = sum(ssims) / len(ssims)

            if ema:
                ema.restore(model)

            # 현재 주기 최고 갱신
            if val_psnr > cycle_best_psnr:
                cycle_best_psnr = val_psnr

            is_best = val_psnr > best_psnr
            if is_best:
                best_psnr = val_psnr
                best_epoch = epoch
                # EMA 가중치로 저장
                if ema:
                    ema.apply_shadow(model)
                torch.save(model.state_dict(), os.path.join(ft_dir, 'best.pth'))
                if ema:
                    ema.restore(model)

            results.append({'epoch': epoch, 'psnr': round(val_psnr, 4),
                            'ssim': round(val_ssim, 4), 'lr': cur_lr})

            print(f"[{epoch:03d}/{ft_config['epochs']}] "
                  f"Loss={train_loss:.4f} | PSNR={val_psnr:.2f}dB | "
                  f"lr={cur_lr:.2e} | {elapsed}s"
                  + ("  ✅ Best!" if is_best else ""))

        # Restart 경계: 주기 얼리스탑 체크
        if epoch in restart_boundaries:
            improve = cycle_best_psnr - prev_cycle_best
            print(f"\n  🔄 Warm Restart @ epoch {epoch}")
            print(f"     이번 주기 best: {cycle_best_psnr:.4f}dB | "
                  f"직전 주기 best: {prev_cycle_best:.4f}dB | "
                  f"개선: {improve:+.4f}dB")

            if prev_cycle_best > 0 and improve < cycle_improve_thresh:
                print(f"  🛑 주기 간 개선 {improve:.4f}dB < {cycle_improve_thresh}dB → 얼리스탑!")
                break
            else:
                print(f"  ✅ 개선 확인 → 다음 주기 계속")
                prev_cycle_best = cycle_best_psnr
                cycle_best_psnr = 0.0  # 다음 주기 초기화

    # 결과 저장
    result = {
        'exp': ft_exp_name,
        'base_exp': exp_name,
        'base_psnr': round(state_psnr, 4),
        'best_psnr': round(best_psnr, 4),
        'improvement': round(best_psnr - state_psnr, 4),
        'best_epoch': best_epoch,
        'epochs_run': epoch,
        'epochs_max': ft_config['epochs'],
        'lr_start': ft_config['lr_start'],
        'T_0': ft_config['T_0'],
        'use_ema': ft_config['use_ema'],
    }
    with open(os.path.join(ft_dir, 'results.json'), 'w') as f:
        json.dump(result, f, indent=2)

    improvement = best_psnr - state_psnr
    print(f"  기존 PSNR: {state_psnr:.4f}dB → 파인튜닝 후: {best_psnr:.4f}dB ({improvement:+.4f}dB)")

    print(f"\n  🏆 {ft_exp_name} 완료 | Best PSNR: {best_psnr:.4f}dB (epoch {best_epoch})")
    return best_psnr


# ── 메인: Top3 순차 실행 ───────────────────────────────────────
if __name__ == '__main__':

    BASE_CONFIG = {
        'exp_root':   '/home/user/experiments',
        'train_data': '/home/user/data/RESIDE/RESIDE-6K/train',
        'val_data':   '/home/user/data/RESIDE/RESIDE-6K/test',
        'patch_size': 256,
        'batch_size': 4,
        'epochs':     200,         # 총 200 epoch
        'lr_start':   5e-5,        # 기존 2e-4의 1/4
        'lr_min':     1e-6,        # 최소 lr
        'T_0':        40,          # 40 epoch마다 Warm Restart
        'T_mult':     2,           # 이후 80 → 160 epoch으로 늘어남
        'use_ema':    True,
    }

    # Top3 대상
    TOP3 = [
        # (실험명, 모델크기, 아키텍처)
        ('GB_0149_B_freq_branch_p256_lr2e-04_L1Perc', 'B', 'freq_branch'),
        ('GA_0065_B_baseline_p256_lr2e-04_L1Perc',    'B', 'baseline'),
        ('GA_0045_S_baseline_p256_lr4e-04_L1Perc',    'S', 'baseline'),
    ]

    print("\n" + "="*65)
    print("  🚀 Top3 Warm Restart 파인튜닝 시작")
    print("  전략: CosineAnnealingWarmRestarts + EMA")
    print(f"  lr: {BASE_CONFIG['lr_start']:.1e} → {BASE_CONFIG['lr_min']:.1e}")
    print(f"  Restart 주기: T_0={BASE_CONFIG['T_0']}, T_mult={BASE_CONFIG['T_mult']}")
    print(f"  예상 소요: ~3~4일 (3개 × 200 epoch)")
    print("="*65)

    final_results = []
    for exp_name, model_size, arch in TOP3:
        psnr = finetune(exp_name, model_size, arch, BASE_CONFIG)
        final_results.append((exp_name, psnr))
        print(f"\n  📊 중간 집계: {exp_name} → {psnr:.4f}dB\n")

    print("\n" + "="*65)
    print("  🎉 Top3 파인튜닝 전체 완료!")
    print("="*65)
    for name, psnr in sorted(final_results, key=lambda x: -x[1]):
        tag = " 👑" if psnr == max(r[1] for r in final_results) else ""
        print(f"  FT_{name}: {psnr:.4f}dB{tag}")

    best_name, best_psnr = max(final_results, key=lambda x: x[1])
    print(f"\n  최종 Best: FT_{best_name} → {best_psnr:.4f}dB")
    if best_psnr >= 30.0:
        print("  🎊 30dB 달성!!")
    else:
        print(f"  30dB까지 {30.0 - best_psnr:.4f}dB 남음 → 앙상블 시도 권장")
