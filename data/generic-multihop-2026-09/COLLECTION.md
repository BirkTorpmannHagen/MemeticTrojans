# Generic-bare multi-hop for non-frontier models (qwen-32B, gemma2-27B, command-r-35B)

**Collected:** 2026-09-21 (queued).
**Collected by:** Claude on birk's behalf.
**Source:** `sandbox/multihop.py` closed loop → qwen2.5:32b, gemma2:27b, command-r:35b via the local
ollama daemon (all reachable).

## Intent
The state-mediated per-surface amplification (`sandbox/reach_by_surface.py`) needs both arms' multi-hop
presence M. The Trojan arm (x_sec_mg/child) existed for all 5 models, but the **generic-bare** arm
(x_sec_gen/bare) existed only for the two frontier models — so the combined table's state-mediated
section covered only gpt-oss & deepseek. This collects the missing generic arm for the other three so
the state-mediated section can extend to all 5.

## Method
`python -m sandbox.multihop --model <m> --tag <tag> --meme x_sec_gen --variant bare --goal none
--hops 4 --n 30` → `out/exposure/multihop_<tag>_x_sec_gen_bare_Gnone.json`. Skip-if-exists; set -uo
(one model failing does not abort). See run.sh.

## Contents
- run.sh, run.log; multihop_<tag>_x_sec_gen_bare_Gnone.json per model (copied on completion).

## Issues
None yet. command-r-35B has no measured install (asr) rate — the reach model uses the documented 0.66
fallback for it; collect asr_model_commandr_35b separately if an exact install rate is needed.

## Issues (update 2026-09-21)
- qwen-32B: collected OK (M_G=1.23). **gemma2-27B and command-r-35B FAILED** — `chat` read-timeout
  after 5 retries (300s/call). These are *local* 27–35B models; the 4-hop×30 closed loop is too
  slow/heavy for the timeout (no fast cloud variant on the daemon). They remain **edge-only** in the
  combined table (edge uses the pre-existing cross_* tx data, not the multi-hop closed loop). To add
  them later: run on a host that serves these models fast, or raise the multihop `chat` timeout and
  expect a multi-hour run.
