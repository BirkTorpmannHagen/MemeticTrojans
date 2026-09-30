#!/usr/bin/env bash
# Local (GPU) crossing collection at goal=none ONLY: base carriers + extra carriers, for the three
# local models. Strictly sequential — one model on the GPU at a time, unload between. Idempotent
# (skip-guarded). qwen first (biggest gap), then gemma2, then command-r. Bash so `set -- $spec`
# word-splits correctly.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
H=http://127.0.0.1:11434

unload_all() {
  for m in $(curl -s "$H/api/ps" | python3 -c "import sys,json;[print(x['name']) for x in json.load(sys.stdin).get('models',[])]" 2>/dev/null); do
    curl -s "$H/api/generate" -d "{\"model\":\"$m\",\"keep_alive\":0}" >/dev/null 2>&1
  done
  sleep 8
}

# case-key(for sweep_crossing.sh)  ollama-model  tag
for spec in "qwen qwen2.5:32b qwen32b" "gemma2 gemma2:27b gemma2_27b" "commandr command-r:35b commandr_35b"; do
  set -- $spec; CASE=$1; MODEL=$2; TAG=$3
  echo "############ $(date -u +%FT%TZ)  LOCAL BLOCK: $MODEL ############"
  unload_all
  echo "=== $(date -u +%FT%TZ) base crossing $MODEL ==="
  bash sandbox/sweep_crossing.sh "$CASE" 2>&1 | tail -3 || true
  echo "=== $(date -u +%FT%TZ) extra crossing $MODEL ==="
  bash sandbox/sweep_crossing_extra.sh "$MODEL" "$TAG" none 2>&1 | grep -aE "^(ok|FAIL|###)" | tail -6 || true
done
echo "########## LOCAL none-only DONE $(date -u +%FT%TZ) ##########"
