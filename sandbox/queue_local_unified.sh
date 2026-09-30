#!/bin/bash
# Queue: wait for the local transmission sweep to exit, THEN derive per-model feed-position
# curves for the four local models (replacing the last gpt-4o-mini dependency). Resumable:
# unified_by_model.py skips already-measured positions.
set -u
cd /Users/birk/Projects/MoltbookContagion

echo "### queued $(date) — waiting for local transmission sweep to end"
while pgrep -f "sweep_content.sh local" >/dev/null 2>&1; do sleep 300; done
echo "### sweep ended $(date) — starting local position-curve assays"

# local models emit JSON directly (not reasoning) -> num_predict 500 is plenty; chat() caps num_ctx=4096
for mt in "gemma2:27b|gemma2_27b" "command-r:35b|commandr_35b" "qwen2.5:32b|qwen32b" "llama3.3:70b|llama33_70b"; do
  model="${mt%%|*}"; tag="${mt##*|}"
  echo "### unified curves ${tag} $(date)"
  python -m sandbox.unified_by_model --model "$model" --tag "$tag" --n 100 --num_predict 500 2>&1 | tail -4
  python -m sandbox.unified_by_model --model "$model" --tag "$tag" --n 100 --mode retransmit 2>&1 | tail -4
done
echo "### LOCAL POSITION CURVES DONE $(date)"

# install ASR (local, K=25) — after transmission sweep + position curves
echo "### LOCAL INSTALL ASR K=25 $(date)"
for mt in "gemma2:27b|gemma2_27b" "command-r:35b|commandr_35b" "qwen2.5:32b|qwen32b" "llama3.3:70b|llama33_70b"; do
  model="${mt%%|*}"; tag="${mt##*|}"
  for meme in security_warning defi_pool continuity airdrop personality; do
    out="gasr_${tag}_${meme}_none.json"
    [ -f "out/exposure/$out" ] && continue
    python -m sandbox.run_asr --backend ollama --model "$model" --meme "$meme" \
       --variants child --goals none --arms ranked --positions 0,5 --n 24 --concurrency 1 --out "$out" 2>&1 | tail -1
  done
done
echo "### LOCAL INSTALL ASR K=25 DONE $(date)"
