#!/usr/bin/env bash
# Local upvote block at n=32 (the 3 local backends). Cloud models stay n=64.
# Sequential on the GPU, unload between models. num_predict 2500 (JSON fits; not the bottleneck).
set -uo pipefail
cd "$(dirname "$0")/../.."; export PYTHONPATH=.
H=http://127.0.0.1:11434
UP="python data/upvote-edge-2026-09/measure_upvote.py"
unload(){ for m in $(python3 -c "import requests;[print(m['name']) for m in requests.get('$H/api/ps').json().get('models',[]) if 'cloud' not in m['name']]"); do curl -s "$H/api/generate" -d "{\"model\":\"$m\",\"keep_alive\":0}">/dev/null 2>&1; done; sleep 8; }
for spec in "qwen2.5:32b qwen32b" "gemma2:27b gemma2_27b" "command-r:35b commandr_35b"; do
  read -r M TAG <<< "$spec"
  echo "### $(date -u +%FT%TZ) LOCAL $M (n=32)"; unload
  $UP --model "$M" --tag "$TAG" --n 32 --num_predict 2500
done
echo "##### LOCAL n=32 DONE $(date -u +%FT%TZ)"
