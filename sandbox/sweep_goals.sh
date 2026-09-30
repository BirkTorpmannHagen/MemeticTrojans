#!/bin/bash
# Comprehensive GOAL x MEME x MODEL sweep for the goal-conditional analysis.
# DECOUPLED design (P(post) is ~meme-independent base activity, and too rare to
# estimate inside the per-meme grid for low-post models like gpt-oss):
#
#   P(post)          per (model, goal)          -> ppost_<tag>_<goal>.json
#                    sampled n=N_POST, np=NP_POST (big enough that reasoning models
#                    emit the action JSON; np too small returns EMPTY -> false 0).
#   P(payload|post)  per (model, goal, meme)     -> gsar_<tag>_<goal>_<meme>.json
#                    prefill only (n_sample=0): fast, unaffected by np truncation.
#   install (ASR)    per (model, goal, meme)     -> gasr_<tag>_<meme>_<goal>.json
#
# SAR_{model,goal,meme} = p_post{model,goal} x p_payload_given_post{model,goal,meme}.
#
# FULLY RESUMABLE: every cell writes its own file and is skipped if present, so a
# laptop close loses at most one cell. Re-launch to continue.
#   bash sandbox/sweep_goals.sh cloud 2>&1 | tee -a /tmp/goalsweep_cloud.log
#   bash sandbox/sweep_goals.sh local 2>&1 | tee -a /tmp/goalsweep_local.log
set -u
cd /Users/birk/Projects/MoltbookContagion
N_POST=128; NP_POST=1200; N_COND=32; ASR_N=24; PPOST_MEME=security_warning
GOALS="none economic degen influencer helpful security_conscious skeptic cautious promotional evangelical curious builder researcher newcomer"
MEMES="security_warning defi_pool airdrop continuity personality"

sweep_model() {
  local model="$1" tag="$2"
  # NOTE: P(post) is collected SEPARATELY by sweep_ppost.sh (needs much larger n for
  # the small create_post rates). This script is prefill P(payload|post) + install only.
  for goal in $GOALS; do
    for meme in $MEMES; do
      # --- P(payload|post): prefill only (fast, correct) ---
      local sar="out/exposure/gsar_${tag}_${goal}_${meme}.json"
      if [ -f "$sar" ]; then echo "### Ppay ${tag}/${goal}/${meme} cached"; else
        echo "### Ppay ${tag}/${goal}/${meme} $(date +%H:%M)"
        python -m sandbox.ollama_sar run --model "$model" --memes "$meme" --goal "$goal" \
           --variant child --n_sample 0 --n_cond $N_COND --out "$sar" 2>&1 | tail -2
      fi
      # --- install (ASR) ---
      local asr_base="gasr_${tag}_${meme}_${goal}.json"
      if [ -f "out/exposure/${asr_base}" ]; then echo "### ASR ${tag}/${meme}/${goal} cached"; else
        echo "### ASR ${tag}/${meme}/${goal} $(date +%H:%M)"
        python -m sandbox.run_asr --backend ollama --model "$model" --meme "$meme" \
           --variants child --goals "$goal" --arms ranked --positions 0,5 --n $ASR_N \
           --concurrency 1 --out "$asr_base" 2>&1 | tail -1
      fi
    done
  done
  echo "### MODEL DONE $tag $(date +%H:%M)"
}

case "${1:-}" in
  cloud)
    sweep_model "gpt-oss:120b-cloud"      gptoss120b
    sweep_model "deepseek-v4-flash:cloud" deepseekflash
    echo "### CLOUD GOAL SWEEP DONE" ;;
  local)
    sweep_model gemma2:27b    gemma2_27b
    sweep_model command-r:35b commandr_35b
    sweep_model qwen2.5:32b   qwen32b
    sweep_model llama3.3:70b  llama33_70b
    echo "### LOCAL GOAL SWEEP DONE" ;;
  *) echo "usage: $0 <cloud|local>"; exit 1 ;;
esac
