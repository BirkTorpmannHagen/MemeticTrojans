#!/usr/bin/env bash
set -u; cd /Users/birk/Projects/MoltbookContagion; export PYTHONPATH=.
run(){ python -m sandbox.ollama_sar run --model "$MODEL" --memes "$1" --goal none --variant "$2" --n_sample 0 --n_cond 40 --semantic --out "out/exposure/$3" 2>&1 | tail -1; }
for spec in "gpt-oss:120b-cloud gptoss120b" "deepseek-v4-flash:cloud deepseekflash"; do
  set -- $spec; MODEL=$1; TAG=$2
  echo "=== $(date -u +%FT%TZ) $MODEL market-tip (memecoin) ==="
  run "x_alpha_af" child  "cross_child_${TAG}_alpha_af.json"
  run "x_alpha_mg" parent "cross_parent_${TAG}_alpha.json"
  run "x_alpha_af" bare   "cross_bare_${TAG}_af.json"
done
echo "ALL DONE $(date -u +%FT%TZ)"
