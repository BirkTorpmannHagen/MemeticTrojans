#!/bin/bash
# Collect command-r install ASR (goal=none, all 5 table memes), then continue the
# normal local transmission sweep. Resumable: cached cells are skipped.
set -u
cd /Users/birk/Projects/MoltbookContagion
MEMES="security_warning defi_pool continuity airdrop personality"  # carrier first

for meme in $MEMES; do
  out="out/exposure/gasr_commandr_35b_${meme}_none.json"
  if [ -f "$out" ]; then echo "### ASR commandr/${meme} cached"; continue; fi
  echo "### ASR commandr/${meme} $(date +%H:%M)"
  python -m sandbox.run_asr --backend ollama --model command-r:35b --meme "$meme" \
     --variants child --goals none --arms ranked --positions 0,5 --n 24 \
     --concurrency 1 --out "gasr_commandr_35b_${meme}_none.json" 2>&1 | tail -1
done
echo "### COMMANDR ASR DONE $(date +%H:%M)"

# continue transmission data collection normally
exec bash sandbox/sweep_content.sh local
