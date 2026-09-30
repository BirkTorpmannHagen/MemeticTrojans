#!/usr/bin/env bash
# Extra-carrier crossing (claw/shell/oclaw/karma + bespoke payloads) for the CLOUD models across
# goal=none + the 6 operator goals. Runs as a BASH script so `set -- $spec` word-splits correctly
# (the inline zsh launches did NOT split, passing an empty tag -> all cells failed silently; this
# fixes that). sweep_crossing_extra.sh now verifies each write and reports FAILS.
#
# SAFETY: simulated LLM assay only; nothing posted to moltbook.com, no skill fetched/executed.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
GOALS="none helpful cautious economic influencer promotional degen"

for spec in "gpt-oss:120b-cloud gptoss120b" "deepseek-v4-flash:cloud deepseekflash"; do
  set -- $spec; MODEL=$1; TAG=$2
  for g in $GOALS; do
    echo "=== $(date -u +%FT%TZ) $MODEL tag=$TAG goal=$g ==="
    bash sandbox/sweep_crossing_extra.sh "$MODEL" "$TAG" "$g" 2>&1
  done
done
echo "########## CLOUD EXTRA (all goals) DONE $(date -u +%FT%TZ) ##########"
