"""
30dB 달성을 위한 순차 파인튜닝 시나리오
우선순위: S1 → S2 → S3 → S4

일정:
  S1: ~8h  (오늘 밤)
  S2: ~6h  (내일 오전)
  S3: ~8h  (내일 오후)
  S4: ~3h  (내일 저녁, 여유 있으면)
  → 월요일부터 앙상블/3DGS

기록: ~/computer_vision/finetune_scenario_log.json
"""

import os, sys, json, time, math
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

LOG_PATH = os.path.join(REPO_DIR, "finetune_scenario_log.json")


# ── EMA ───────────────────────────────────────────────────────
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = self.decay * self.shadow[n] + (1 - self.decay) * p.data

    def apply_shadow(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.data.clone()
                p.data = self.shadow[n]

    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data = self.backup[n]


# ── 로그 관리 ─────────────────────────────────────────────────
def load_log():
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            return json.load(f)
    return {"scenarios": [], "global_best": {"psnr": 29.6715, "exp": "FT_GB_0149", "updated": ""}}


def save_log(log):
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


def log_scenario(log, scenario_id, name, base_psnr, result_psnr, epochs_run, notes=""):
    improvement = result_psnr - base_psnr
    entry = {
        "scenario_id": scenario_id,
        "name": name,
        "base_psnr": round(base_psnr, 4),
        "result_psnr": round(result_psnr, 4),
        "improvement": round(improvement, 4),
        "epochs_run": epochs_run,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "done",
        "notes": notes,
        "reached_30db": result_psnr >= 30.0,
    }
    log["scenarios"].append(entry)
    if result_psnr > log["global_best"]["psnr"]:
        log["global_best"] = {
            "psnr": round(result_psnr, 4),
            "exp": name,
            "updated": entry["timestamp"]
        }
    save_log(log)
    return entry


# ── 공통 학습 함수 ─────────────────────────────────────────────
def run_finetune(cfg):
    device = torch.device("cuda")
    exp_dir = os.path.join(cfg["exp_root"], cfg["exp_name"])
    os.makedirs(exp_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  🔧 [{cfg['scenario_id']}] {cfg['exp_name']}")
    print(f"  기반: {cfg['base_exp']} ({cfg['base_psnr']:.4f}dB)")
    print(f"  lr: {cfg['lr_start']:.1e}→{cfg['lr_min']:.1e} | patch: {cfg['patch_size']} | epochs: {cfg['epochs']}")
    print(f"  전략: {cfg['strategy_note']}")
    print(f"{'='*65}\n")

    # 모델 로드
    model = build_model(size=cfg["model_size"]).to(device)
    ckpt = torch.load(cfg["ckpt_path"], map_location=device, weights_only=False)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    print(f"  ✅ 로드 완료: {cfg['ckpt_path']}")

    train_loader = get_dataloader(cfg["train_data"], patch_size=cfg["patch_size"],
                                  batch_size=cfg["batch_size"], is_train=True)
    val_loader   = get_dataloader(cfg["val_data"],   patch_size=cfg["patch_size"],
                                  batch_size=1, is_train=False)

    criterion = DehazeFormerLoss(
        l1_w=cfg.get("l1_w", 1.0), perceptual_w=cfg.get("perc_w", 0.1),
        ssim_w=0.0, freq_w=0.0
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr_start"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg["T_0"], T_mult=cfg.get("T_mult", 2), eta_min=cfg["lr_min"]
    )
    scaler = GradScaler("cuda")
    ema    = EMA(model, decay=0.9995) if cfg.get("use_ema", True) else None

    best_psnr  = cfg["base_psnr"]
    best_epoch = 0

    # Restart 경계 계산
    restart_boundaries = []
    t, ep = cfg["T_0"], 0
    T_mult = cfg.get("T_mult", 2)
    while ep + t <= cfg["epochs"]:
        ep += t
        restart_boundaries.append(ep)
        t *= T_mult
    print(f"  Restart 경계: {restart_boundaries}\n")

    cycle_best = 0.0
    prev_cycle_best = 0.0
    improve_thresh = 0.003

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        train_loss, t0 = 0.0, time.time()

        for hazy, clear, _ in train_loader:
            hazy, clear = hazy.to(device), clear.to(device)
            optimizer.zero_grad()
            with autocast("cuda"):
                pred = model(hazy)
                loss = criterion(pred, clear)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            if ema: ema.update(model)

        scheduler.step()
        train_loss /= len(train_loader)
        elapsed = int(time.time() - t0)
        cur_lr  = optimizer.param_groups[0]["lr"]

        if math.isnan(train_loss) or math.isinf(train_loss):
            print(f"  ❌ NaN @ epoch {epoch} → 중단")
            break

        if epoch % 5 == 0:
            if ema: ema.apply_shadow(model)
            model.eval()
            psnrs = []
            with torch.no_grad():
                for hazy, clear, _ in val_loader:
                    hazy, clear = hazy.to(device), clear.to(device)
                    with autocast("cuda"):
                        pred = model(hazy).clamp(0, 1)
                    psnrs.append(compute_psnr(pred, clear))
            val_psnr = sum(psnrs) / len(psnrs)
            if ema: ema.restore(model)

            if val_psnr > cycle_best:
                cycle_best = val_psnr

            is_best = val_psnr > best_psnr
            if is_best:
                best_psnr  = val_psnr
                best_epoch = epoch
                if ema: ema.apply_shadow(model)
                torch.save(model.state_dict(), os.path.join(exp_dir, "best.pth"))
                if ema: ema.restore(model)

            # 결과 즉시 저장 (중단돼도 기록 남음)
            with open(os.path.join(exp_dir, "progress.json"), "w") as f:
                json.dump({"current_epoch": epoch, "current_psnr": round(val_psnr, 4),
                           "best_psnr": round(best_psnr, 4), "best_epoch": best_epoch}, f)

            tag = "  ✅ Best!" if is_best else ""
            star = " 🌟 30dB!!" if val_psnr >= 30.0 else ""
            print(f"[{epoch:03d}/{cfg['epochs']}] Loss={train_loss:.4f} | "
                  f"PSNR={val_psnr:.2f}dB | lr={cur_lr:.2e} | {elapsed}s{tag}{star}")

            if val_psnr >= 30.0:
                print(f"\n  🎊 30dB 달성!! ({val_psnr:.4f}dB) → 저장 후 계속")

        if epoch in restart_boundaries:
            improve = cycle_best - prev_cycle_best
            print(f"\n  🔄 Restart @ ep{epoch} | 이번:{cycle_best:.4f} 직전:{prev_cycle_best:.4f} 개선:{improve:+.4f}dB")
            if prev_cycle_best > 0 and improve < improve_thresh:
                print(f"  🛑 개선 {improve:.4f}dB < {improve_thresh}dB → 얼리스탑")
                break
            prev_cycle_best = cycle_best
            cycle_best = 0.0

    result_psnr = best_psnr
    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump({
            "scenario_id": cfg["scenario_id"], "exp": cfg["exp_name"],
            "base_exp": cfg["base_exp"], "base_psnr": cfg["base_psnr"],
            "best_psnr": round(result_psnr, 4),
            "improvement": round(result_psnr - cfg["base_psnr"], 4),
            "best_epoch": best_epoch, "epochs_run": epoch,
            "strategy": cfg["strategy_note"],
        }, f, indent=2)

    print(f"\n  🏁 완료: {cfg['base_psnr']:.4f} → {result_psnr:.4f}dB "
          f"({result_psnr - cfg['base_psnr']:+.4f}dB)\n")
    return result_psnr, epoch


# ── 시나리오 정의 ─────────────────────────────────────────────
EXP_ROOT   = "/home/user/experiments"
TRAIN_DATA = "/home/user/data/RESIDE/RESIDE-6K/train"
VAL_DATA   = "/home/user/data/RESIDE/RESIDE-6K/test"

SCENARIOS = [
    # S1: FT_GB_0149 연장 (ep200에서 아직 오르던 중 → 계속 진행 여지 있음)
    {
        "scenario_id": "S1",
        "exp_name":    "S1_GB0149_extend_lr1e5",
        "base_exp":    "FT_GB_0149_B_freq_branch_p256_lr2e-04_L1Perc",
        "base_psnr":   29.6715,
        "ckpt_path":   f"{EXP_ROOT}/FT_GB_0149_B_freq_branch_p256_lr2e-04_L1Perc/best.pth",
        "model_size":  "B",
        "exp_root":    EXP_ROOT,
        "train_data":  TRAIN_DATA, "val_data": VAL_DATA,
        "patch_size":  256, "batch_size": 4,
        "epochs":      100,
        "lr_start":    1e-5,
        "lr_min":      1e-7,
        "T_0":         30, "T_mult": 2,
        "use_ema":     True,
        "strategy_note": "GB_0149 연장: lr 1e-5로 낮춰 100 epoch 추가, ep200 미수렴 이어서",
    },
    # S2: patch 320 재학습 (이미지 400×400 → 최대 safe patch)
    {
        "scenario_id": "S2",
        "exp_name":    "S2_GB0149_patch320",
        "base_exp":    "FT_GB_0149_B_freq_branch_p256_lr2e-04_L1Perc",
        "base_psnr":   29.6715,
        "ckpt_path":   f"{EXP_ROOT}/FT_GB_0149_B_freq_branch_p256_lr2e-04_L1Perc/best.pth",
        "model_size":  "B",
        "exp_root":    EXP_ROOT,
        "train_data":  TRAIN_DATA, "val_data": VAL_DATA,
        "patch_size":  320, "batch_size": 3,  # patch 320, batch 3
        "epochs":      80,
        "lr_start":    2e-5,
        "lr_min":      1e-7,
        "T_0":         25, "T_mult": 2,
        "use_ema":     True,
        "strategy_note": "patch 256→320(이미지 400 한계): 더 넓은 맥락 학습, batch 3",
    },
    # S3: S1 결과 기반 추가 연장
    {
        "scenario_id": "S3",
        "exp_name":    "S3_best_extend_lr5e6",
        "base_exp":    "S1_GB0149_extend_lr1e5",
        "base_psnr":   29.7567,
        "ckpt_path":   f"{EXP_ROOT}/S1_GB0149_extend_lr1e5/best.pth",
        "model_size":  "B",
        "exp_root":    EXP_ROOT,
        "train_data":  TRAIN_DATA, "val_data": VAL_DATA,
        "patch_size":  256, "batch_size": 4,
        "epochs":      80,
        "lr_start":    5e-6,
        "lr_min":      1e-8,
        "T_0":         25, "T_mult": 2,
        "use_ema":     True,
        "strategy_note": "S1 이후 극소 lr(5e-6) 80 epoch 마무리 정밀 조정",
    },
    # S4: Perceptual 비중 낮춘 L1-heavy
    {
        "scenario_id": "S4",
        "exp_name":    "S4_GB0149_L1heavy",
        "base_exp":    "FT_GB_0149_B_freq_branch_p256_lr2e-04_L1Perc",
        "base_psnr":   29.6715,
        "ckpt_path":   f"{EXP_ROOT}/FT_GB_0149_B_freq_branch_p256_lr2e-04_L1Perc/best.pth",
        "model_size":  "B",
        "exp_root":    EXP_ROOT,
        "train_data":  TRAIN_DATA, "val_data": VAL_DATA,
        "patch_size":  256, "batch_size": 4,
        "epochs":      80,
        "lr_start":    1e-5,
        "lr_min":      1e-7,
        "T_0":         25, "T_mult": 2,
        "l1_w":        1.0, "perc_w": 0.02,
        "use_ema":     True,
        "strategy_note": "Perceptual 비중 0.1→0.02, L1 집중으로 PSNR 직접 최적화",
    },
]


# ── 메인 실행 ─────────────────────────────────────────────────
if __name__ == "__main__":
    log = load_log()

    # 이미 완료된 시나리오 스킵
    done_ids = {s["scenario_id"] for s in log["scenarios"] if s.get("status") in ("done", "skipped")}

    print("\n" + "="*65)
    print("  🎯 30dB 달성 시나리오 순차 실행")
    print(f"  현재 Global Best: {log['global_best']['psnr']}dB ({log['global_best']['exp']})")
    print(f"  실행할 시나리오: {[s['scenario_id'] for s in SCENARIOS if s['scenario_id'] not in done_ids]}")
    print("="*65)

    for cfg in SCENARIOS:
        sid = cfg["scenario_id"]
        if sid in done_ids:
            print(f"\n  ⏭️  {sid} 이미 완료, 스킵")
            continue

        # S3는 S1 결과로 base_psnr 업데이트
        if sid == "S3":
            s1_result = next((s for s in log["scenarios"] if s["scenario_id"] == "S1"), None)
            if s1_result:
                cfg["base_psnr"] = s1_result["result_psnr"]
                print(f"\n  S3 base_psnr → S1 결과 {cfg['base_psnr']}dB로 업데이트")

        result_psnr, epochs_run = run_finetune(cfg)
        entry = log_scenario(log, sid, cfg["exp_name"], cfg["base_psnr"],
                             result_psnr, epochs_run, cfg["strategy_note"])

        print(f"\n  📊 [{sid}] {cfg['exp_name']}")
        print(f"     {entry['base_psnr']}dB → {entry['result_psnr']}dB ({entry['improvement']:+.4f}dB)")
        print(f"  🏆 현재 Global Best: {log['global_best']['psnr']}dB\n")

        if result_psnr >= 30.0:
            print("  🎊 30dB 달성!! 이후 시나리오 계속 실행해도 됨")

    print("\n" + "="*65)
    print("  ✅ 전체 파인튜닝 시나리오 완료")
    print(f"  최종 Global Best: {log['global_best']['psnr']}dB ({log['global_best']['exp']})")
    print("  → 다음 단계: 앙상블 / 3DGS preprocessing")
    print("="*65)
