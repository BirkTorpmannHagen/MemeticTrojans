#!/usr/bin/env bash
# Per-meme P(upvote) + P(retransmit) for CURRENT models (gpt-oss, deepseek) via ollama cloud.
# run_unified measures the upvote + payload_tx channels per meme in one pass (rank 0, goal none).
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
DIR=data/meme-upvote-cloud-2026-09
MEMES="security_warning continuity airdrop personality defi_pool"
for spec in "gpt-oss:120b-cloud gptoss120b" "deepseek-v4-flash:cloud deepseekflash"; do
  set -- $spec; MODEL=$1; TAG=$2
  for MEME in $MEMES; do
    OUTF="out/exposure/unified_permeme_${TAG}_${MEME}.json"
    if [ -f "$OUTF" ]; then echo "skip $TAG/$MEME (done)"; continue; fi
    echo "=== $(date -u +%FT%TZ)  $MODEL / $MEME ==="
    python -m sandbox.run_unified --backend ollama --model "$MODEL" \
      --n 80 --positions 0 --meme "$MEME" --goals none --primes off --incentives off \
      --variant child --concurrency 4 --out "unified_permeme_${TAG}_${MEME}.json" \
      && python -c "import shutil; shutil.copy('$OUTF','$DIR/')" 2>/dev/null || echo "  (run failed for $TAG/$MEME, continuing)"
  done
done
echo "ALL DONE"
