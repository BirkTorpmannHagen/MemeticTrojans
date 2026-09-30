#!/usr/bin/env bash
# clawtasks carrier — bare/parent/child assay (cross_* files) for gpt-oss + deepseek, ollama cloud.
# Mirrors sweep_crossing.sh / sweep_bespoke.sh for the single new carrier `claw` (payload cv=claw-verify).
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
NC=40
run(){ # meme variant outbase
  local out="out/exposure/$3"
  [ -f "$out" ] && { echo "### $3 cached"; return; }
  echo "### $3 $(date -u +%FT%TZ)"
  python -m sandbox.ollama_sar run --model "$MODEL" --memes "$1" --goal none \
    --variant "$2" --n_sample 0 --n_cond "$NC" --semantic --out "$out" 2>&1 | tail -1
}
for spec in "gpt-oss:120b-cloud gptoss120b" "deepseek-v4-flash:cloud deepseekflash"; do
  set -- $spec; MODEL=$1; TAG=$2
  echo "=== $(date -u +%FT%TZ) $MODEL clawtasks carrier ==="
  run "x_claw_cv" child  "cross_child_${TAG}_claw_cv.json"   # Trojan: clawtasks carrier + claw-verify
  run "x_claw_mg" parent "cross_parent_${TAG}_claw.json"     # pure contagion: carrier idea only
  run "x_claw_cv" bare   "cross_bare_${TAG}_cv.json"         # payload-alone margin (claw-verify, no carrier)
  cp out/exposure/cross_child_${TAG}_claw_cv.json out/exposure/cross_parent_${TAG}_claw.json \
     data/clawtasks-carrier-cloud-2026-09/ 2>/dev/null || true
done
echo "ALL DONE $(date -u +%FT%TZ)"
