#!/usr/bin/env bash
# Per-model feed-rank curve collection (gpt-oss-120B, deepseek-v4-flash) via ollama cloud.
# Run from repo root:  bash data/rank-curves-cloud-2026-09/run.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
DIR=data/rank-curves-cloud-2026-09
POS=0,2,5,8,11,14,17,20,24
N=80

for spec in "gpt-oss:120b-cloud gptoss120b" "deepseek-v4-flash:cloud deepseekflash"; do
  set -- $spec; MODEL=$1; TAG=$2
  echo "=== $(date -u +%FT%TZ)  $MODEL -> unified_${TAG}.json ==="
  python -m sandbox.run_unified --backend ollama --model "$MODEL" \
    --n "$N" --positions "$POS" --meme security_warning \
    --goals none --primes off --incentives off --variant child \
    --concurrency 4 --out "unified_${TAG}.json"
  cp "out/exposure/unified_${TAG}.json" "$DIR/unified_${TAG}.json"
  echo "=== $(date -u +%FT%TZ)  done $TAG ==="
done
echo "ALL DONE"
