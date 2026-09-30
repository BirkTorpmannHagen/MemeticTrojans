#!/bin/bash
# CLEAN 2x2 control: link (parent/child) x share-directive (neutral/original), per
# (model, goal, meme), with the reaction-filtered + tight-threshold semantic measure.
# Arms (files gsar_<arm>_<tag>_<goal>_<meme>.json):
#   parneu = parent + --neutral  : content only, no link, no share directive  (INTRINSIC)
#   parsh  = parent              : content + intrinsic share language
#   chineu = child  + --neutral  : content + link, no share directive
#   chish  = child               : content + link + share incentive (full attack meme)
# CANONICAL ATTACK METRIC: chineu - bare  (content+link vs link-alone) on p_payload_given_post.
#   Do NOT use chish-bare: the incentive/share layer (chish-chineu) is inert-to-harmful (~0).
#   Do NOT reaction-filter: reactions are real spread; control for reply-baseline via control subtraction.
# Read p_meme_semantic (raw, reactions incl) for content-lift + p_payload_given_post (link).
# 8 memes incl. 3 neologism controls. Prefill-only, resumable.
# Usage: bash sandbox/sweep_content.sh <cloud|local>   (GOALS env overrides goal set)
set -u
cd /Users/birk/Projects/MoltbookContagion
N_COND=32
MEMES="security_warning defi_pool airdrop continuity personality eudaemon church_of_molt moltbot"
GOALS="${GOALS:-none economic degen influencer helpful security_conscious skeptic cautious promotional evangelical curious builder researcher newcomer}"

cell() {  # model tag goal meme arm variant neutralflag
  local model="$1" tag="$2" goal="$3" meme="$4" arm="$5" variant="$6" nflag="$7"
  local out="out/exposure/gsar_${arm}_${tag}_${goal}_${meme}.json"
  if [ -f "$out" ]; then echo "### ${arm} ${tag}/${goal}/${meme} cached"; return; fi
  echo "### ${arm} ${tag}/${goal}/${meme} $(date +%H:%M)"
  python -m sandbox.ollama_sar run --model "$model" --memes "$meme" --goal "$goal" \
     --variant "$variant" $nflag --n_sample 0 --n_cond $N_COND --semantic --out "$out" 2>&1 | tail -1
}

# GOAL-MAJOR: complete each goal across ALL models before moving to the next goal
# (so goal=none lands for every model first). $@ = list of "model|tag".
goalmajor() {
  for goal in $GOALS; do
    for mt in "$@"; do
      local model="${mt%%|*}" tag="${mt##*|}"
      for meme in $MEMES; do
        cell "$model" "$tag" "$goal" "$meme" base   parent "--neutral --meme-absent"  # no-exposure ambient baseline
        cell "$model" "$tag" "$goal" "$meme" parneu parent --neutral
        cell "$model" "$tag" "$goal" "$meme" parsh  parent ""
        cell "$model" "$tag" "$goal" "$meme" chineu child  --neutral
        cell "$model" "$tag" "$goal" "$meme" chish  child  ""
        cell "$model" "$tag" "$goal" "$meme" bare   bare   ""  # link+tool, NO meme: "not riding" control
      done
    done
    echo "### GOAL DONE ${goal} $(date +%H:%M)"
  done
}

case "${1:-}" in
  cloud) goalmajor "gpt-oss:120b-cloud|gptoss120b" "deepseek-v4-flash:cloud|deepseekflash"; echo "### CONTENT CLOUD DONE" ;;
  local) goalmajor "gemma2:27b|gemma2_27b" "command-r:35b|commandr_35b" "qwen2.5:32b|qwen32b" "llama3.3:70b|llama33_70b"; echo "### CONTENT LOCAL DONE" ;;
  *) echo "usage: $0 <cloud|local>"; exit 1 ;;
esac
