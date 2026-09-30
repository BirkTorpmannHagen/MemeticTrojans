#!/bin/bash
# Reliable per-meme SAR + per-meme ASR (install) sweep across models.
# Higher n than the initial local_families run (which returned n_post=0 for gemma2).
# Resumable: skips a cell whose output json already exists. Local models serialize
# on the GPU, so they run sequentially; each model does all 5 memes.
#
#   SAR  -> out/exposure/sar_sweep_<tag>.json           (all 5 memes, one call)
#   ASR  -> out/exposure/asr_sweep_<tag>_<meme>.json    (per meme, positions 0 & 5)
#
# Usage: bash sandbox/sweep_sar_asr.sh <local|cloud> 2>&1 | tee /tmp/sweep.log
set -u
cd /Users/birk/Projects/MoltbookContagion
MEMES="security_warning,defi_pool,continuity,airdrop,personality"
MEME_LIST="security_warning defi_pool continuity airdrop personality"
NSAMPLE=160; NCOND=48; ASR_N=48

sweep_model() {
  M=$1; TAG=$2
  local sar="out/exposure/sar_sweep_${TAG}.json"
  if [ -f "$sar" ]; then
    echo "### SAR $M -> cached ($sar)"
  else
    echo "### SAR $M (5 memes, n_sample=$NSAMPLE) $(date +%H:%M)"
    python -m sandbox.ollama_sar run --model "$M" --memes "$MEMES" --goal none \
       --variant child --n_sample $NSAMPLE --n_cond $NCOND --out "$sar" 2>&1 | tail -9
  fi
  for meme in $MEME_LIST; do
    local asr_base="asr_sweep_${TAG}_${meme}.json"   # run_asr prepends out/exposure/
    local asr="out/exposure/${asr_base}"
    if [ -f "$asr" ]; then echo "### ASR $M/$meme -> cached"; continue; fi
    echo "### ASR $M/$meme (n=$ASR_N, pos 0,5) $(date +%H:%M)"
    python -m sandbox.run_asr --backend ollama --model "$M" --meme "$meme" \
       --variants child --goals none --arms ranked --positions 0,5 --n $ASR_N \
       --concurrency 1 --out "$asr_base" 2>&1 | tail -2
  done
  echo "### DONE $M $(date +%H:%M)"
}

case "${1:-local}" in
  local)
    sweep_model gemma2:27b    gemma2_27b
    sweep_model command-r:35b commandr_35b
    sweep_model qwen2.5:32b   qwen32b
    sweep_model llama3.3:70b  llama33_70b     # 70B last (slowest)
    echo "### LOCAL SWEEP DONE" ;;
  cloud)
    sweep_model "$CLOUD_GPTOSS"   gptoss120b
    sweep_model "$CLOUD_DEEPSEEK" deepseekflash
    echo "### CLOUD SWEEP DONE" ;;
esac
