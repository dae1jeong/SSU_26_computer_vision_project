"""
실험 자동 스케줄러
configs/ 폴더의 모든 실험을 순서대로 실행
GPU 상태 체크 → 학습 → 결과 기록 → GitHub push
"""

import os
import sys
import glob
import json
import time
import subprocess
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train import train


def get_gpu_memory_free():
    """여유 GPU 메모리(MB) 반환"""
    if not torch.cuda.is_available():
        return 0
    torch.cuda.empty_cache()
    free = torch.cuda.mem_get_info()[0] // (1024 ** 2)
    return free


def git_push(message):
    """결과를 GitHub에 push"""
    try:
        subprocess.run(['git', 'add', '-A'], cwd='/home/user/computer_vision', check=True)
        subprocess.run(['git', 'commit', '-m', message], cwd='/home/user/computer_vision', check=True)
        subprocess.run(['git', 'push'], cwd='/home/user/computer_vision', check=True)
        print(f"  📤 GitHub push: {message}")
    except Exception as e:
        print(f"  ⚠️  Git push 실패: {e}")


def run_all_experiments(config_dir='configs', min_free_mb=4000):
    configs = sorted(glob.glob(os.path.join(config_dir, 'exp*.json')))
    total   = len(configs)
    print(f"\n{'='*60}")
    print(f"  총 {total}개 실험 시작")
    print(f"{'='*60}\n")

    summary = []

    for idx, cfg_path in enumerate(configs):
        with open(cfg_path) as f:
            config = json.load(f)

        exp_name = config['exp_name']
        exp_dir  = os.path.join(config['exp_root'], exp_name)

        # 이미 완료된 실험 스킵
        if os.path.exists(os.path.join(exp_dir, 'results.json')):
            print(f"[{idx+1:03d}/{total}] ⏭  Skip (already done): {exp_name}")
            continue

        # GPU 여유 확인
        while get_gpu_memory_free() < min_free_mb:
            print(f"  ⏳ GPU 메모리 부족 ({get_gpu_memory_free()}MB). 5분 대기...")
            time.sleep(300)

        print(f"\n[{idx+1:03d}/{total}] 🚀 Start: {exp_name}")
        t0 = time.time()

        try:
            best_psnr = train(config)
            elapsed   = (time.time() - t0) / 60
            row = {'exp': exp_name, 'best_psnr': best_psnr, 'time_min': f"{elapsed:.1f}"}
            summary.append(row)
            print(f"  ✅ Done in {elapsed:.1f}min | Best PSNR={best_psnr:.2f}dB")

            # 10개마다 GitHub push
            if (idx + 1) % 10 == 0:
                git_push(f"결과 업데이트: {idx+1}/{total}개 완료")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            row = {'exp': exp_name, 'best_psnr': -1, 'error': str(e)}
            summary.append(row)

    # 전체 결과 저장
    with open('experiments/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    git_push(f"🎉 전체 실험 완료: {total}개")
    print(f"\n{'='*60}")
    print(f"  모든 실험 완료!")
    top5 = sorted([r for r in summary if r['best_psnr'] > 0],
                  key=lambda x: x['best_psnr'], reverse=True)[:5]
    print(f"  Top 5 결과:")
    for r in top5:
        print(f"    {r['exp']}: {r['best_psnr']:.2f}dB")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    os.makedirs('experiments', exist_ok=True)
    run_all_experiments()
