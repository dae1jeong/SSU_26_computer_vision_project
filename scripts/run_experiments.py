"""
실험 자동 스케줄러 (GPU 서버용)
- Early Stopping 적용
- 5개마다 텔레그램 알림
- GitHub 자동 push
"""

import os
import sys
import glob
import json
import time
import subprocess
import torch
import requests

TELEGRAM_TOKEN = None   # 텔레그램은 OpenClaw가 처리
GITHUB_REMOTE  = os.environ.get("GITHUB_REMOTE", "origin")
REPO_DIR       = "/home/user/computer_vision"

sys.path.append(REPO_DIR)
from scripts.train import train


def get_gpu_free_mb():
    if not torch.cuda.is_available():
        return 0
    torch.cuda.empty_cache()
    return torch.cuda.mem_get_info()[0] // (1024 ** 2)


def git_push(message):
    try:
        subprocess.run(['git', '-C', REPO_DIR, 'add', '-A'], check=True, capture_output=True)
        result = subprocess.run(['git', '-C', REPO_DIR, 'commit', '-m', message],
                                capture_output=True, text=True)
        if 'nothing to commit' not in result.stdout:
            subprocess.run(['git', '-C', REPO_DIR, 'push', GITHUB_REMOTE, 'main'],
                           check=True, capture_output=True)
            print(f"  📤 GitHub push: {message}")
    except Exception as e:
        print(f"  ⚠️  Git push 실패: {e}")


def load_summary():
    path = os.path.join(REPO_DIR, 'experiments', 'summary.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def save_summary(summary):
    os.makedirs(os.path.join(REPO_DIR, 'experiments'), exist_ok=True)
    with open(os.path.join(REPO_DIR, 'experiments', 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


def run_all(config_dir=None, min_free_mb=3000, notify_every=5):
    if config_dir is None:
        config_dir = os.path.join(REPO_DIR, 'configs')

    configs = sorted(glob.glob(os.path.join(config_dir, 'G*.json')))
    total   = len(configs)
    summary = load_summary()
    done_exps = {r['exp'] for r in summary}

    print(f"\n{'='*60}")
    print(f"  총 {total}개 실험 | 이미 완료: {len(done_exps)}개")
    print(f"{'='*60}\n")

    completed = len(done_exps)
    best_ever = max((r['best_psnr'] for r in summary if r.get('best_psnr', 0) > 0), default=0)

    for idx, cfg_path in enumerate(configs):
        with open(cfg_path) as f:
            config = json.load(f)

        exp_name = config['exp_name']
        if exp_name in done_exps:
            continue

        # GPU 여유 대기
        while get_gpu_free_mb() < min_free_mb:
            print(f"  ⏳ GPU 부족 ({get_gpu_free_mb()}MB). 3분 대기...")
            time.sleep(180)

        print(f"\n[{completed+1}/{total}] 🚀 {exp_name}")
        t0 = time.time()

        try:
            best_psnr = train(config)
            elapsed   = (time.time() - t0) / 60
            row = {'exp': exp_name, 'best_psnr': best_psnr,
                   'time_min': round(elapsed, 1), 'config': cfg_path}
            summary.append(row)
            done_exps.add(exp_name)
            completed += 1

            if best_psnr > best_ever:
                best_ever = best_psnr
                print(f"  🏆 새 최고 기록! PSNR={best_ever:.2f}dB")

            # 5개마다 알림 파일 작성 (OpenClaw heartbeat가 읽음)
            if completed % notify_every == 0:
                top3 = sorted([r for r in summary if r.get('best_psnr', 0) > 0],
                              key=lambda x: x['best_psnr'], reverse=True)[:3]
                msg = (f"✅ 실험 {completed}/{total}개 완료\n"
                       f"🏆 현재 Best PSNR: {best_ever:.2f}dB\n"
                       f"Top 3:\n" +
                       "\n".join(f"  {r['exp']}: {r['best_psnr']:.2f}dB" for r in top3))
                with open('/home/user/notify.txt', 'w') as f:
                    f.write(msg)
                git_push(f"실험 {completed}/{total} 완료 | Best={best_ever:.2f}dB")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            summary.append({'exp': exp_name, 'best_psnr': -1, 'error': str(e)})

        save_summary(summary)

    # 최종 완료
    top5 = sorted([r for r in summary if r.get('best_psnr', 0) > 0],
                  key=lambda x: x['best_psnr'], reverse=True)[:5]
    final_msg = (f"🎉 전체 실험 완료! ({total}개)\n"
                 f"🏆 Best PSNR: {best_ever:.2f}dB\n"
                 f"Top 5:\n" +
                 "\n".join(f"  {r['exp']}: {r['best_psnr']:.2f}dB" for r in top5))
    with open('/home/user/notify.txt', 'w') as f:
        f.write(final_msg)
    git_push("🎉 전체 실험 완료!")
    print(f"\n{final_msg}")


if __name__ == '__main__':
    run_all()
