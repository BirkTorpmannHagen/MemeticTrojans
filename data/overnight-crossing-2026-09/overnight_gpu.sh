#!/usr/bin/env bash
# Overnight LOCAL/GPU collection: finish the remaining multihop, then extend the carrier x payload
# crossing (goal=none) and the goal-resolved crossing across the 6 operator goals, for the two local
# models (command-r:35b, gemma2:27b). STRICTLY SEQUENTIAL — one model on the GPU at a time. Each big
# model keeps the GPU for its whole block (multihop + base + all goals) so we reload at most once per
# model. Idempotent: every cell/file is skip-guarded, so re-running just fills gaps.
#
# Concurrency: the sibling overnight_cloud.sh runs the cloud models at the same time (no GPU
# contention), and the windowed-top snapshotter (network only) also runs alongside — all fine.
#
# SAFETY: pure simulated LLM assays via ollama (sandbox.ollama_sar / sandbox.multihop). Nothing is
# posted to moltbook.com; no skill is fetched or executed.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
H=http://127.0.0.1:11434
GOALS="helpful cautious economic influencer promotional degen"

unload_all() {   # evict every resident ollama model so the next gets full VRAM
  for m in $(curl -s "$H/api/ps" | python3 -c "import sys,json;[print(x['name']) for x in json.load(sys.stdin).get('models',[])]" 2>/dev/null); do
    curl -s "$H/api/generate" -d "{\"model\":\"$m\",\"keep_alive\":0}" >/dev/null 2>&1
  done
  sleep 8
}

block() {  # $1=ollama model  $2=case-key for sweep_crossing.sh  $3=tag
  local MODEL="$1" CASE="$2" TAG="$3"
  echo "############ $(date -u +%FT%TZ)  MODEL BLOCK: $MODEL ############"
  # multihop (only x_sec_gen bare exists in the pipeline; skip if already collected)
  local MH="out/exposure/multihop_${TAG}_x_sec_gen_bare_Gnone.json"
  if [ ! -f "$MH" ]; then
    echo "=== $(date -u +%FT%TZ) multihop $MODEL ==="
    python -m sandbox.multihop --model "$MODEL" --tag "$TAG" --meme x_sec_gen --variant bare \
      --goal none --hops 4 --n 30 2>&1 | tail -2 || true
  else
    echo "skip multihop $TAG (done)"
  fi
  # base crossing (goal=none)
  echo "=== $(date -u +%FT%TZ) base crossing $MODEL ==="
  bash sandbox/sweep_crossing.sh "$CASE" 2>&1 | tail -3 || true
  # goal-resolved crossing
  for g in $GOALS; do
    echo "=== $(date -u +%FT%TZ) crossing goal=$g $MODEL ==="
    bash sandbox/sweep_crossing_goal.sh "$MODEL" "$TAG" "$g" 2>&1 | tail -3 || true
  done
  echo "############ $(date -u +%FT%TZ)  DONE BLOCK: $MODEL ############"
}

# command-r is already resident from the prior job -> do it first WITHOUT an initial unload.
block "command-r:35b" commandr commandr_35b
# switch models: evict command-r, then gemma2 gets the whole GPU.
unload_all
block "gemma2:27b" gemma2 gemma2_27b

echo "########## GPU ORCHESTRATOR DONE $(date -u +%FT%TZ) ##########"
