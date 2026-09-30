#!/usr/bin/env bash
# Confirmatory re-estimate of each backend's EXPLORATORY-BEST carrier (max r_up in
# data/overnight-crossing-2026-09, n_cond=32, seed0=1000) plus its generic-sharer floor, on FRESH
# independent samples (seed0=20000 -> disjoint personas and feeds). Removes the winner's-curse bias
# of plugging the selected carrier's exploratory estimate into Tables 1/A.
#
# Cloud cells (gpt-oss, deepseek) run concurrently in the background (off-box inference, no GPU);
# local cells run strictly sequentially, one model on the GPU at a time. Idempotent (skip-guarded).
#
# SAFETY: pure simulated LLM assays via ollama (sandbox.ollama_sar). Nothing is posted to
# moltbook.com; no skill is fetched or executed.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
H=http://127.0.0.1:11434
SEED0=20000
OUT=out/exposure
DIR=data/confirmatory-bestcarrier-2026-09

run() { # model meme variant n_cond outfile
  local out="$OUT/$5"
  [ -f "$out" ] && { echo "### $5 cached"; return; }
  echo "### $(date -u +%FT%TZ) start $5 (model=$1 meme=$2 variant=$3 n_cond=$4 seed0=$SEED0)"
  python -m sandbox.ollama_sar run --model "$1" --memes "$2" --goal none --variant "$3" \
     --n_sample 0 --n_cond "$4" --seed0 "$SEED0" --semantic --out "$out" 2>&1 | tail -1
  if [ -f "$out" ]; then echo "ok $5 $(date -u +%FT%TZ)"; else echo "FAIL $5 $(date -u +%FT%TZ)"; fi
}

unload_local() {   # unload resident LOCAL models only (never touch the cloud-proxied ones)
  for m in $(curl -s "$H/api/ps" | python3 -c "import sys,json;[print(x['name']) for x in json.load(sys.stdin).get('models',[])]" 2>/dev/null | grep -v cloud); do
    curl -s "$H/api/generate" -d "{\"model\":\"$m\",\"keep_alive\":0}" >/dev/null 2>&1
  done
  sleep 8
}

cloud_block() {
  run gpt-oss:120b-cloud      x_sec_mg   child 128 confirm_child_gptoss120b_sec_mg.json
  run gpt-oss:120b-cloud      x_sec_gen  bare  128 confirm_bare_gptoss120b_gen.json
  run deepseek-v4-flash:cloud x_consc_ss child 128 confirm_child_deepseekflash_consc_ss.json
  run deepseek-v4-flash:cloud x_sec_gen  bare  128 confirm_bare_deepseekflash_gen.json
  echo "##### CLOUD DONE $(date -u +%FT%TZ)"
}

local_block() {
  # ollama-model  tag  best-carrier-meme  cell
  for spec in "qwen2.5:32b qwen32b x_consc_ss consc_ss" \
              "gemma2:27b gemma2_27b x_sec_mg sec_mg" \
              "command-r:35b commandr_35b x_sec_mg sec_mg"; do
    set -- $spec; M=$1; TAG=$2; MEME=$3; CELL=$4
    echo "############ $(date -u +%FT%TZ)  LOCAL: $M ############"
    unload_local
    run "$M" "$MEME"   child 128 "confirm_child_${TAG}_${CELL}.json"
    run "$M" x_sec_gen bare  64  "confirm_bare_${TAG}_gen.json"
  done
  echo "##### LOCAL DONE $(date -u +%FT%TZ)"
}

cloud_block > "$DIR/cloud.log" 2>&1 &
CPID=$!
local_block
wait "$CPID"
cp "$OUT"/confirm_*.json "$DIR"/ 2>/dev/null || true      # provenance copy next to the docs
echo "########## CONFIRMATORY DONE $(date -u +%FT%TZ) ##########"
