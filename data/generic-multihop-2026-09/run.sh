#!/usr/bin/env bash
# Generic-bare multi-hop (x_sec_gen / bare) closed loop for the 3 non-frontier models, so the
# state-mediated per-surface amplification (sandbox.reach_by_surface) can extend to all 5 models.
# Trojan multihop (x_sec_mg/child) already exists for these; only the generic arm was missing.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
DIR=data/generic-multihop-2026-09
for spec in "qwen2.5:32b qwen32b" "gemma2:27b gemma2_27b" "command-r:35b commandr_35b"; do
  set -- $spec; MODEL=$1; TAG=$2
  OUTF="out/exposure/multihop_${TAG}_x_sec_gen_bare_Gnone.json"
  if [ -f "$OUTF" ]; then echo "skip $TAG (done)"; continue; fi
  echo "=== $(date -u +%FT%TZ)  $MODEL generic-bare multihop ==="
  python -m sandbox.multihop --model "$MODEL" --tag "$TAG" --meme x_sec_gen --variant bare \
    --goal none --hops 4 --n 30 2>&1 | tail -2
  [ -f "$OUTF" ] && cp "$OUTF" "$DIR/" || echo "  (no output for $TAG)"
done
echo "ALL DONE"
