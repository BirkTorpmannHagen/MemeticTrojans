# Finish gemma2:27b + command-r:35b (clean-feed action-dist + generic multi-hop)

**Collected:** 2026-09 (local).
**Collected by:** Claude on birk's behalf.
**Source:** `data/finish-gemma-commandr-2026-09/run.sh` → `sandbox/action_dist.py` (clean-feed,
meme-absent) then `sandbox/multihop.py` (generic-bare closed loop), for gemma2:27b and command-r:35b
via the local ollama daemon.

## Intent
Fill the two local backends that had failed earlier due to VRAM contention (a resident model with
`keep_alive` pinned blocked the next load). Completes the per-model clean-feed action distribution
(`action_dist_clean_*`) and the generic-bare multi-hop needed alongside the frontier models.

## Method
Strictly sequential — one model on the GPU at a time, unloading the resident model before each load
(the KEY FIX vs the prior failures). See `run.sh`; `run.log` is the run transcript.

## Contents
- `run.sh`, `run.log` — driver + transcript.
- `action_dist_clean_gemma2_27b.json` — clean-feed action distribution (command-r written to
  `out/exposure/` alongside).
- `multihop_gemma2_27b_x_sec_gen_bare_Gnone.json` — generic-bare multi-hop result.
- Primary outputs also land in `out/exposure/` (shared assay tree).

## Issues
None known so far.
