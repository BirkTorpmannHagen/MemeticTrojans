#!/usr/bin/env bash
# Measured upvote edge (see measure_upvote.py): 10 carrier cells + 1 no-stimulus control, n=64 each,
# seed0=30000, `new` feed, rank 0. Cloud (gpt-oss) runs now; local models wait until the GPU is free
# (confirmatory local block done, or no local model loaded), then run one at a time.
# deepseek-v4-flash is NOT run: retired on ollama cloud 2026-09-25.
# SAFETY: simulated assay only; nothing is posted to moltbook.com, no skill is fetched.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
H=http://127.0.0.1:11434
DIR=data/upvote-edge-2026-09
CONF_LOG=data/confirmatory-bestcarrier-2026-09/run.log
PY="python $DIR/measure_upvote.py"

n_local_loaded() {
  python3 -c "import requests;print(sum('cloud' not in m['name'] for m in requests.get('$H/api/ps').json().get('models',[])))"
}
unload_local() {
  for m in $(python3 -c "import requests;[print(m['name']) for m in requests.get('$H/api/ps').json().get('models',[]) if 'cloud' not in m['name']]"); do
    curl -s "$H/api/generate" -d "{\"model\":\"$m\",\"keep_alive\":0}" >/dev/null 2>&1
  done
  sleep 8
}

cloud_block() {
  echo "### $(date -u +%FT%TZ) gpt-oss start"
  $PY --model gpt-oss:120b-cloud --tag gptoss120b --num_predict 2000
  echo "##### CLOUD DONE $(date -u +%FT%TZ)"
}

local_block() {
  until grep -q "LOCAL DONE" "$CONF_LOG" 2>/dev/null || [ "$(n_local_loaded)" = "0" ]; do sleep 120; done
  echo "### $(date -u +%FT%TZ) GPU free, starting local models"
  for spec in "qwen2.5:32b qwen32b" "gemma2:27b gemma2_27b" "command-r:35b commandr_35b"; do
    read -r M TAG <<< "$spec"
    echo "############ $(date -u +%FT%TZ) LOCAL: $M ############"
    unload_local
    $PY --model "$M" --tag "$TAG" --num_predict 400
  done
  echo "##### LOCAL DONE $(date -u +%FT%TZ)"
}

cloud_block > "$DIR/cloud.log" 2>&1 &
CPID=$!
local_block > "$DIR/local.log" 2>&1
wait "$CPID"
echo "########## UPVOTE-EDGE DONE $(date -u +%FT%TZ) ##########"
