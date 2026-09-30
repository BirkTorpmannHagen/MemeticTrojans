#!/usr/bin/env bash
# Finish gemma2:27b + command-r:35b (local) — clean-feed action-dist THEN generic multi-hop.
# KEY FIX: unload the resident model before each local model so it gets the whole GPU (the prior
# failures were VRAM contention: keep_alive pinned the previous model, blocking the next load).
# Strictly SEQUENTIAL — one model on the GPU at a time.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
DIR=data/finish-gemma-commandr-2026-09
H=http://127.0.0.1:11434

unload_all() {   # evict every resident model so the next gets full VRAM
  for m in $(curl -s "$H/api/ps" | python3 -c "import sys,json;[print(x['name']) for x in json.load(sys.stdin).get('models',[])]" 2>/dev/null); do
    curl -s "$H/api/generate" -d "{\"model\":\"$m\",\"keep_alive\":0}" >/dev/null 2>&1
  done
  sleep 8
}

echo "### PHASE 1: clean-feed action-dist (N=100) ###"
for spec in "gemma2:27b gemma2_27b" "command-r:35b commandr_35b"; do
  set -- $spec; MODEL=$1; TAG=$2
  OUTF="out/exposure/action_dist_clean_${TAG}.json"
  [ -f "$OUTF" ] && { echo "skip clean $TAG (done)"; continue; }
  echo "=== $(date -u +%FT%TZ) clean-feed $MODEL ==="; unload_all
  python -m sandbox.action_dist --model "$MODEL" --n 100 --meme-absent 2>&1 | tail -2
  [ -f "$OUTF" ] && cp "$OUTF" "$DIR/" || echo "  (no output $TAG)"
done

echo "### PHASE 2: generic-bare multi-hop ###"
for spec in "gemma2:27b gemma2_27b" "command-r:35b commandr_35b"; do
  set -- $spec; MODEL=$1; TAG=$2
  OUTF="out/exposure/multihop_${TAG}_x_sec_gen_bare_Gnone.json"
  [ -f "$OUTF" ] && { echo "skip multihop $TAG (done)"; continue; }
  echo "=== $(date -u +%FT%TZ) multihop $MODEL ==="; unload_all
  python -m sandbox.multihop --model "$MODEL" --tag "$TAG" --meme x_sec_gen --variant bare \
    --goal none --hops 4 --n 30 2>&1 | tail -2
  [ -f "$OUTF" ] && cp "$OUTF" "$DIR/" || echo "  (no output $TAG)"
done
echo "### ALL DONE $(date -u +%FT%TZ) ###"
