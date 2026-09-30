# Prefill SAR by feed rank (gpt-oss-120B, deepseek-v4-flash) — cloud

**Collected:** 2026-09-16 (queued)
**Collected by:** Claude on birk's behalf
**Source:** `sandbox/ollama_sar.py run` (prefill decomposition) → ollama.com cloud
`gpt-oss:120b-cloud`, `deepseek-v4-flash:cloud`, via the local signed-in ollama daemon.

## Intent
Measure how the **retransmission rate** — SAR = `P(post) · P(payload | post)`, i.e. the probability
an agent authors a NEW payload-carrying post — depends on the **feed rank** at which the agent saw
the payload. The reach sim currently uses a rank-0 scalar SAR (prefill was only ever measured at
rank 0); this sweeps rank so `p_ret` can be rank-resolved (or confirmed rank-independent). See
memory `retransmission-newpost-prefill` and `docs/SIMULATION_MODEL.md` §4/§10.

## Method
`ollama_sar run` with a new `--positions` sweep. Per (rank): `P(post)` = P(a `create_post` appears)
sampled over `n_sample=120` natural heartbeats; `P(payload|post)` via **prefill** (force the response
to begin a `create_post`, autocomplete, check payload markers) over `n_cond=50`. SAR = product.

- meme = `security_warning`, variant = `child`, goal = `none` (matches the exposure collection)
- positions (feed ranks) = **0, 5, 11, 17, 24**
- command (see `run.sh`):
  `python -m sandbox.ollama_sar run --model <cloud-model> --memes security_warning --goal none
   --variant child --n_sample 120 --n_cond 50 --positions 0,5,11,17,24 --out sar_by_rank_<tag>.json`

Outputs → `out/exposure/sar_by_rank_<tag>.json` (rows tagged with `position`), copied here.

## Contents
- `run.sh` — launch script; `run.log` — stdout (per-rank P(post)/P(payload|post)/SAR).
- `sar_by_rank_gptoss120b.json`, `sar_by_rank_deepseekflash.json` — copied on completion. Each row:
  position, p_post, n_post, p_payload_given_post (natural if ≥5 posts else prefill),
  prefill_p_payload_given_post, sar_payload_permille, etc.

## Issues
- 2026-09-16: first launch **failed mid-run** (a transient DNS blip: `lookup ollama.com: no such
  host`) after gpt-oss ranks 0,5; `set -e` killed the script before anything was saved. Root cause:
  `ollama_sar.chat` treated the daemon's cloud-proxy network error as fatal. Fixed by adding
  "no such host"/"dial tcp"/"connection refused/reset"/"no route to host"/"eof" to the retryable
  set, then relaunched. No partial data kept.
- 2026-09-20: second launch — gpt-oss finished all 5 ranks and saved, but `run.sh` passed `--out
  sar_by_rank_<tag>.json` (relative) which the runner uses verbatim → the file landed at the repo
  ROOT, not `out/exposure/`; the `cp` then failed and `set -e` killed the script before deepseek.
  Fixed: `--out` now full `out/exposure/...` path + `run.sh` skips a model whose output already
  exists (so relaunch only collects the missing deepseek half). gpt-oss file rescued (moved to
  `out/exposure/`), no data lost. gpt-oss rank-resolved result: SAR_pay decays 18‰(r0)→0.3‰(r24),
  P(payload|post) 0.54→0.04 — retransmission is concentrated at the top of the feed.
- Watch: gpt-oss `P(post)` is ~2.5% → ~3 natural posts at n_sample=120, so its SAR is
`P(post)`-limited and noisy (the conditional falls back to prefill); deepseek posts often (~37%) so
its P(post) is well-sampled. Prefill `P(payload|post)` is a forced-post conditional → an upper
estimate of the natural rate. Reasoning model (gpt-oss) is slow per call.
