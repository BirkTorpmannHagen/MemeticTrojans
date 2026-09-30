#!/bin/bash
# Experiment 1: carrier x payload crossing. child cells + carrier(parent) & payload(bare) margins.
# ollama_sar run --semantic reports p_meme_semantic (carrier onward tx) + p_payload_given_post (payload tx).
set -u
cd /Users/birk/Projects/MoltbookContagion
NC=32
run() { # meme variant outbase
  local out="out/exposure/$3"
  [ -f "$out" ] && { echo "### $3 cached"; return; }
  echo "### $3 $(date +%H:%M)"
  python -m sandbox.ollama_sar run --model "$MODEL" --memes "$1" --goal none \
     --variant "$2" --n_sample 0 --n_cond $NC --semantic --out "$out" 2>&1 | tail -1
}
sweep() {
  MODEL="$1"; local tag="$2"
  for c in sec defi cont neutral molt verify econ auton consc alpha; do
    for p in mg esc sk sfx; do run "x_${c}_${p}" child "cross_child_${tag}_${c}_${p}.json"; done
  done
  for c in sec defi cont neutral molt verify econ auton consc alpha; do run "x_${c}_mg" parent "cross_parent_${tag}_${c}.json"; done   # carrier-only
  for p in mg esc sk sfx; do run "x_sec_${p}" bare "cross_bare_${tag}_${p}.json"; done               # payload-only
  echo "### CROSSING DONE ${tag} $(date +%H:%M)"
}
case "$1" in
  gptoss)   sweep "gpt-oss:120b-cloud" gptoss120b ;;
  deepseek) sweep "deepseek-v4-flash:cloud" deepseekflash ;;
  gemma2)   sweep "gemma2:27b" gemma2_27b ;;
  commandr) sweep "command-r:35b" commandr_35b ;;
  qwen)     sweep "qwen2.5:32b" qwen32b ;;
esac
