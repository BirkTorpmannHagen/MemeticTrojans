# Per-meme P(upvote) + P(retransmit), current models (gpt-oss, deepseek) — cloud

**Collected:** 2026-09-20 (queued).
**Collected by:** Claude on birk's behalf.
**Source:** `sandbox/run_unified.py` (E5 unified heartbeat) → `gpt-oss:120b-cloud`,
`deepseek-v4-flash:cloud` via the local signed-in ollama daemon.

## Intent
The per-meme carrier trade-off (P(upvote) vs P(retransmit)) previously had both channels only from
gpt-4o-mini, which is **quarantined**. This collects both channels for the **current** models so the
trade-off scatter uses gpt-oss / deepseek instead. (qwen sampled_sar already has per-meme
retransmission but no upvote; gpt-oss/deepseek had only security_warning.)

## Method
`run_unified` at **rank 0**, goal=none, prime off, incentive off, variant=child, **n=80**, one meme
per run, over the canonical 5-meme set: security_warning, continuity, airdrop, personality, defi_pool.
It reports, from one heartbeat, `upvote` (P(upvote)) and `payload_tx` (P(retransmit), any re-emission)
plus install / payload_post etc. Skip-if-output-exists; `set -uo` (not -e) so one meme's failure does
not abort the sweep. See `run.sh`; outputs → `out/exposure/unified_permeme_<tag>_<meme>.json`, copied here.

## Contents
- `run.sh`, `run.log`; `unified_permeme_<tag>_<meme>.json` per (model, meme) on completion.
- Consumed by `analysis/meme_carrier_tradeoff.py` (updated to prefer these current-model files over gpt-4o-mini).

## Issues
None yet. Watch: gpt-oss is a rare poster, so `payload_tx`/`payload_post` are low-count at n=80 (upvote
is well-sampled); reasoning model is slow per call. Cloud blips now retry (ollama_sar.chat hardening;
run_unified uses its own backend — watch for network errors in run.log).
