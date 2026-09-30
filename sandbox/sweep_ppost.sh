#!/bin/bash
# SEPARATE P(post) collection (distinct from the prefill P(payload|post) grid).
# ADAPTIVE N: create_post rates are small (Moltbook's skill.md steers agents to
# comment, not broadcast — gpt-oss is a genuine rare poster, ~0 in 100 draws), so a
# flat n=100 can't resolve a ~0.5% rate. Escalate n=100 -> 1000 -> 3000 per (model,
# goal), stopping as soon as a tier has >= RELIABLE posts (enough for a stable rate).
# Each tier writes its own file; the highest-n present is the estimate. Meme-pooled
# via one representative feed. FULLY RESUMABLE (per-tier files skipped if present).
#   bash sandbox/sweep_ppost.sh cloud 2>&1 | tee -a /tmp/ppost_cloud.log
#   bash sandbox/sweep_ppost.sh local 2>&1 | tee -a /tmp/ppost_local.log
set -u
cd /Users/birk/Projects/MoltbookContagion
TIERS="100 1000 3000"; RELIABLE=5; NP=1200; PPOST_MEME=security_warning
GOALS="none economic degen influencer helpful security_conscious skeptic cautious promotional evangelical curious builder researcher newcomer"

nfield() { python -c "import json,sys; print(json.load(open(sys.argv[1]))[0]['n_post'])" "$1" 2>/dev/null || echo 0; }

ppost_model() {
  local model="$1" tag="$2"
  for goal in $GOALS; do
    local done=0
    for N in $TIERS; do
      local f="out/exposure/ppost_${tag}_${goal}_n${N}.json"
      if [ ! -f "$f" ]; then
        echo "### P(post) n=$N ${tag}/${goal} $(date +%H:%M)"
        python -m sandbox.ollama_sar run --model "$model" --memes "$PPOST_MEME" --goal "$goal" \
           --variant child --n_sample $N --n_cond 0 --np_post $NP --out "$f" 2>&1 | tail -2
      fi
      local n; n=$(nfield "$f")
      if [ "$n" -ge "$RELIABLE" ]; then
        echo "### ${tag}/${goal} reliable at n=$N (n_post=$n) -> p_post=$(python -c "import json;print(round(json.load(open('$f'))[0]['p_post'],4))")"
        done=1; break
      fi
    done
    [ "$done" = 0 ] && echo "### ${tag}/${goal} still < $RELIABLE posts at n=$(echo $TIERS|awk '{print $NF}') — rate is sub-resolution (upper-bounded)"
  done
  echo "### PPOST MODEL DONE $tag $(date +%H:%M)"
}

case "${1:-}" in
  cloud)
    ppost_model "gpt-oss:120b-cloud"      gptoss120b
    ppost_model "deepseek-v4-flash:cloud" deepseekflash
    echo "### PPOST CLOUD DONE" ;;
  local)
    ppost_model gemma2:27b    gemma2_27b
    ppost_model command-r:35b commandr_35b
    ppost_model qwen2.5:32b   qwen32b
    ppost_model llama3.3:70b  llama33_70b
    echo "### PPOST LOCAL DONE" ;;
  *) echo "usage: $0 <cloud|local>"; exit 1 ;;
esac
