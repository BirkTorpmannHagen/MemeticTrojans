#!/usr/bin/env bash
# Prefill SAR by feed rank: P(post) x P(payload|post) at each rank, gpt-oss + deepseek, ollama cloud.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
DIR=data/sar-by-rank-cloud-2026-09
POS=0,5,11,17,24
for spec in "gpt-oss:120b-cloud gptoss120b" "deepseek-v4-flash:cloud deepseekflash"; do
  set -- $spec; MODEL=$1; TAG=$2
  if [ -f "out/exposure/sar_by_rank_${TAG}.json" ]; then
    echo "=== $(date -u +%FT%TZ)  skip $TAG (already collected) ==="; continue
  fi
  echo "=== $(date -u +%FT%TZ)  $MODEL SAR-by-rank ==="
  python -m sandbox.ollama_sar run --model "$MODEL" --memes security_warning --goal none \
    --variant child --n_sample 120 --n_cond 50 --positions "$POS" \
    --out "out/exposure/sar_by_rank_${TAG}.json"          # full path: runner uses --out verbatim
  cp "out/exposure/sar_by_rank_${TAG}.json" "$DIR/sar_by_rank_${TAG}.json"
  echo "=== $(date -u +%FT%TZ)  done $TAG ==="
done
echo "ALL DONE"
