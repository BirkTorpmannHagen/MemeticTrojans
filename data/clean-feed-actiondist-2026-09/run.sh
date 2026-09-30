#!/usr/bin/env bash
# Clean-feed (no-meme) action distribution at N=100 per model — the ambient/baseline action rates
# to compare against the meme-present action table. sandbox.action_dist --meme-absent.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
DIR=data/clean-feed-actiondist-2026-09
for MODEL in "gpt-oss:120b-cloud" "deepseek-v4-flash:cloud" "qwen2.5:32b" "gemma2:27b" "command-r:35b"; do
  TAG=$(echo "$MODEL" | sed 's/[:/]/_/g')
  OUTF="out/exposure/action_dist_clean_${TAG}.json"
  if [ -f "$OUTF" ]; then echo "skip $MODEL (done)"; continue; fi
  echo "=== $(date -u +%FT%TZ)  $MODEL clean-feed N=100 ==="
  python -m sandbox.action_dist --model "$MODEL" --n 100 --meme-absent 2>&1 | tail -3
  [ -f "$OUTF" ] && cp "$OUTF" "$DIR/" || echo "  (no output for $MODEL)"
done
echo "ALL DONE"
