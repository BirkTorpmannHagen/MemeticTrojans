#!/usr/bin/env bash
# Item 4(a): give shell/oclaw/karma their OWN bespoke payloads and fix claw (claw_cv, not claw_mg).
# Collects the 4 corrected carriers for BOTH channels, all 5 models:
#   * upvote r_up  -> measure_upvote.py --cells (installs table; generic denominator reused from the
#                     main upvote-edge-2026-09 run, so that must be finished first)
#   * payload      -> sandbox.ollama_sar cross_child (fig_bare_vs_trojan; claw_cv already exists, skipped)
# deepseek backend = deepseek-v4.1-flash. Cloud concurrent; local sequential w/ GPU unload.
# SAFETY: simulated assay only; nothing posted to moltbook.com, no skill fetched.
set -uo pipefail
cd "$(dirname "$0")/../.."; export PYTHONPATH=.
H=http://127.0.0.1:11434
UP="python data/upvote-edge-2026-09/measure_upvote.py"
CELLS="claw:x_claw_cv:child,shell:x_shell_sf:child,oclaw:x_oclaw_ok:child,karma:x_karma_ke:child"
# parent = carrier text with NO payload link (the pure-contagion upvote rate); distinct P-suffixed cells
PCELLS="secP:x_sec_mg:parent,clawP:x_claw_cv:parent,shellP:x_shell_sf:parent,oclawP:x_oclaw_ok:parent,karmaP:x_karma_ke:parent,moltP:x_molt_mt:parent,econP:x_econ_ch:parent,autonP:x_auton_sc:parent,conscP:x_consc_ss:parent,alphaP:x_alpha_af:parent"
PAY_CELLS="shell_sf oclaw_ok karma_ke"        # claw_cv payload already collected

unload_local(){ for m in $(python3 -c "import requests;[print(m['name']) for m in requests.get('$H/api/ps').json().get('models',[]) if 'cloud' not in m['name']]"); do curl -s "$H/api/generate" -d "{\"model\":\"$m\",\"keep_alive\":0}">/dev/null 2>&1; done; sleep 8; }

pay(){ # model tag cell npred
  local out="out/exposure/cross_child_$2_$3.json"; [ -f "$out" ] && { echo "cached $out"; return; }
  python -m sandbox.ollama_sar run --model "$1" --memes "x_$3" --goal none --variant child \
     --n_sample 0 --n_cond "$4" --seed0 21000 --semantic --out "$out" 2>&1 | tail -1
}

cloud_block(){
  for CS in "$CELLS" "$PCELLS"; do
    $UP --model gpt-oss:120b-cloud      --tag gptoss120b     --num_predict 6000 --cells "$CS"
    $UP --model deepseek-v4.1-flash:cloud --tag deepseek41flash --num_predict 3000 --cells "$CS"
  done
  for c in $PAY_CELLS; do pay gpt-oss:120b-cloud gptoss120b "$c" 128; pay deepseek-v4.1-flash:cloud deepseek41flash "$c" 128; done
  echo "##### BESPOKE CLOUD DONE $(date -u +%FT%TZ)"
}
local_block(){
  for spec in "qwen2.5:32b qwen32b" "gemma2:27b gemma2_27b" "command-r:35b commandr_35b"; do
    read -r M TAG <<< "$spec"; unload_local
    $UP --model "$M" --tag "$TAG" --n 32 --num_predict 2500 --cells "$CELLS"
    $UP --model "$M" --tag "$TAG" --n 32 --num_predict 2500 --cells "$PCELLS"
    for c in $PAY_CELLS; do pay "$M" "$TAG" "$c" 32; done
  done
  echo "##### BESPOKE LOCAL DONE $(date -u +%FT%TZ)"
}
MODE="${1:-all}"
case "$MODE" in
  cloud) cloud_block > data/bespoke-fix-2026-09/cloud.log 2>&1 ;;
  local) local_block > data/bespoke-fix-2026-09/local.log 2>&1 ;;
  all)   cloud_block > data/bespoke-fix-2026-09/cloud.log 2>&1 & CPID=$!
         local_block > data/bespoke-fix-2026-09/local.log 2>&1; wait "$CPID" ;;
esac
echo "########## BESPOKE-FIX ($MODE) DONE $(date -u +%FT%TZ) ##########"
