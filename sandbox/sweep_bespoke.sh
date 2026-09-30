#!/bin/bash
# Bespoke matched payloads for the 5 data-driven carriers (each carrier + its OWN themed skill
# link, so bare/Trojan echo the carrier's own meme, not the shared security molt-guard), plus a
# generic link-sharer FLOOR (same molt-guard link, description stripped). Removes the
# description-echo confound in the bare baseline. sec/defi/cont already have matched pairs
# (mg/esc/sk) so are NOT re-run here.
#   child -> cross_child_<tag>_<c>_<pk>.json     bare -> cross_bare_<tag>_<pk>.json
#   generic floor -> cross_bare_<tag>_gen.json
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
  # carrier tag : bespoke payload tag
  for pair in "molt:mt" "econ:ch" "auton:sc" "consc:ss" "alpha:af"; do
    c="${pair%%:*}"; pk="${pair##*:}"
    run "x_${c}_${pk}" child "cross_child_${tag}_${c}_${pk}.json"   # Trojan (carrier + own payload)
    run "x_${c}_${pk}" bare  "cross_bare_${tag}_${pk}.json"          # bare (own payload alone)
  done
  run "x_sec_gen" bare "cross_bare_${tag}_gen.json"                  # generic-sharer floor
  echo "### BESPOKE DONE ${tag} $(date +%H:%M)"
}
case "$1" in
  gptoss)   sweep "gpt-oss:120b-cloud" gptoss120b ;;
  deepseek) sweep "deepseek-v4-flash:cloud" deepseekflash ;;
  gemma2)   sweep "gemma2:27b" gemma2_27b ;;
  commandr) sweep "command-r:35b" commandr_35b ;;
esac
