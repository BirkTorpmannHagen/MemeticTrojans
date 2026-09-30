#!/usr/bin/env bash
# Extra INDEPENDENT rephrasing chains for the two cloud backends, to give the depth-trend test real
# replication (chain is the statistical unit; see sandbox/multihop_depth_trend.py). Cloud-only, so it does
# not touch the local GPU. The two backends hit different endpoints, so they run concurrently (6 workers
# each, matching the original working config). collect.py is resumable per chain (checkpoints each depth).
# Usage: bash data/multihop-rup-2026-09/run_cloud_chains.sh [N_chains=7]
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
COL=data/multihop-rup-2026-09/collect.py
N="${1:-7}"
CFG="--meme x_sec_mg --hops 4 --n_author 30 --n_judge 48 --np_judge 3500 --workers 6"

do_backend() {  # model tag
  for c in $(seq 1 "$N"); do
    echo "=== $(date +%H:%M) $2 chain $c/$N ==="
    python "$COL" --model "$1" --tag "$2" --chain "$c" $CFG
  done
  echo "=== $(date +%H:%M) $2 ALL $N CHAINS DONE ==="
}

do_backend "gpt-oss:120b-cloud"        gptoss120b    &
do_backend "deepseek-v4.1-flash:cloud" deepseekflash &
wait
echo "=== $(date +%H:%M) ALL CLOUD CHAINS DONE ==="
