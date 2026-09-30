#!/usr/bin/env bash
# r_up per hop, security carrier (x_sec_mg), Trojan (child) + pure-carrier (parent) arms, 5 backends.
# Cloud first (concurrent), then local models serially with a GPU unload between them.
# Usage: bash data/multihop-rup-2026-09/run.sh
set -u
cd "$(dirname "$0")/../.."            # repo root
export PYTHONPATH=.
COL=data/multihop-rup-2026-09/collect.py
HOPS=4; NA=30; NJ=48

run() {  # model tag np_judge workers  (child cascade; payload+ vs carrier-only split is per-post)
  echo "=== $(date +%H:%M) $2 ==="
  python "$COL" --model "$1" --tag "$2" --meme x_sec_mg \
    --hops $HOPS --n_author $NA --n_judge $NJ --np_judge "$3" --workers "$4"
}

# cloud and local hit different hardware, so they can run concurrently: launch this script twice,
# `run.sh cloud` and `run.sh local`, in parallel. `run.sh` (no arg) runs both sequentially.
TARGET="${1:-all}"

do_cloud() {  # gpt-oss needs a large num_predict to avoid action-block truncation
  run "gpt-oss:120b-cloud"        gptoss120b    3500 6
  run "deepseek-v4.1-flash:cloud" deepseekflash 3500 6
  echo "=== $(date +%H:%M) CLOUD DONE ==="
}

do_local() {  # shared GPU: run serially, unload between models
  run "qwen2.5:32b"    qwen32b      2500 3
  ollama stop "qwen2.5:32b"   2>/dev/null || true
  run "gemma2:27b"     gemma2_27b   2500 3
  ollama stop "gemma2:27b"    2>/dev/null || true
  run "command-r:35b"  commandr_35b 2500 3
  ollama stop "command-r:35b" 2>/dev/null || true
  echo "=== $(date +%H:%M) LOCAL DONE ==="
}

do_local_light() {  # deadline config: lighter judging (np 1000, n_judge 24) -- ~2.5x faster on GPU.
  # collect.py is resumable (checkpoints each depth, skips completed models), so this is safe to re-run.
  local L="--meme x_sec_mg --hops 4 --n_author 30 --n_judge 24 --np_judge 1000 --workers 4"
  echo "=== $(date +%H:%M) qwen32b ===";      python "$COL" --model "qwen2.5:32b"   --tag qwen32b      $L; ollama stop "qwen2.5:32b"   2>/dev/null || true
  echo "=== $(date +%H:%M) gemma2_27b ===";   python "$COL" --model "gemma2:27b"    --tag gemma2_27b   $L; ollama stop "gemma2:27b"    2>/dev/null || true
  echo "=== $(date +%H:%M) commandr_35b ==="; python "$COL" --model "command-r:35b" --tag commandr_35b $L; ollama stop "command-r:35b" 2>/dev/null || true
  echo "=== $(date +%H:%M) LOCAL_LIGHT DONE ==="
}

case "$TARGET" in
  cloud)       do_cloud ;;
  local)       do_local ;;
  local_light) do_local_light ;;
  *)           do_cloud; do_local ;;
esac
echo "=== $(date +%H:%M) $TARGET DONE ==="
