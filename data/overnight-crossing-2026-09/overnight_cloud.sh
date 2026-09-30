#!/usr/bin/env bash
# Overnight CLOUD collection: extend the carrier x payload crossing (goal=none) and the goal-resolved
# crossing across the 6 operator goals, for the two cloud models (gpt-oss:120b-cloud,
# deepseek-v4-flash:cloud) via the local ollama cloud proxy. Runs CONCURRENTLY with overnight_gpu.sh
# (cloud inference is off-box, so no GPU contention). Idempotent: every cell is skip-guarded.
#
# SAFETY: pure simulated LLM assays via ollama (sandbox.ollama_sar). Nothing is posted to
# moltbook.com; no skill is fetched or executed.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
GOALS="helpful cautious economic influencer promotional degen"

# model-ollama-name  case-key(for sweep_crossing.sh)  tag
CLOUD=("gpt-oss:120b-cloud gptoss gptoss120b" "deepseek-v4-flash:cloud deepseek deepseekflash")

# base crossing (goal=none) for both cloud models first
for spec in "${CLOUD[@]}"; do
  set -- $spec; MODEL=$1; CASE=$2; TAG=$3
  echo "=== $(date -u +%FT%TZ) base crossing $MODEL ==="
  bash sandbox/sweep_crossing.sh "$CASE" 2>&1 | tail -3 || true
done

# goal-resolved crossing across all goals for both cloud models
for g in $GOALS; do
  for spec in "${CLOUD[@]}"; do
    set -- $spec; MODEL=$1; CASE=$2; TAG=$3
    echo "=== $(date -u +%FT%TZ) crossing goal=$g $MODEL ==="
    bash sandbox/sweep_crossing_goal.sh "$MODEL" "$TAG" "$g" 2>&1 | tail -3 || true
  done
done

echo "########## CLOUD ORCHESTRATOR DONE $(date -u +%FT%TZ) ##########"
