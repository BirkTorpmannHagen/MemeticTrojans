# Measured upvote edge of the carriers (vs generic Moltbook posts)

**Collected:** 2026-09-26 (launched).
**Collected by:** Claude on birk's behalf.
**Source:** `data/upvote-edge-2026-09/run.sh` → `measure_upvote.py` (natural heartbeat assay, reusing
`sandbox.run_asr`/`sandbox.unified_by_model`/`sandbox.ollama_sar` helpers), local ollama daemon
(gpt-oss-120B and deepseek-v4.1-flash via ollama cloud; qwen2.5-32B, gemma2-27B, command-r-35B on
the local GPU).

## Intent
The installs model's feed-entry probability `p_break = P(r_up·U ≥ τ_s)·θ_s` scales the empirical
upvote distribution U of real Moltbook posts by the carrier's upvote edge `r_up`. `r_up` was a proxy
(`reaction_frac`: regex share of prefilled continuations that refer to the source post, over the bare
link post). This measures it from **explicit upvote actions**, normalised by the upvote rate of
**ordinary Moltbook posts** (user decision, 2026-09-26), for input to a threshold-crossing model of
upvote accrual.

## Method
- Natural heartbeats (no prefill), temperature 1, goal=none, prime off, incentive off. Persona i and
  feed seed `30000+i` (disjoint from crossing 1000.. and confirmatory 20000..), **paired across cells**.
- Feed: K=25 real corpus posts from a random 6 h window, ranked **newest first** (`feed_sort="new"`,
  lead line rewritten to "newest first"); stimulus at **rank 0** with that slot's upvote count.
- Cells (n=64 each): the 10 carriers of `expected_installs_surface.CARRIERS` (Trojan child posts:
  sec_mg, claw_mg, shell_mg, oclaw_mg, karma_mg, molt_mt, econ_ch, auton_sc, consc_ss, alpha_af);
  `generic` = the DENOMINATOR, measured exactly like a Trojan: per trial one uniformly sampled real
  corpus post (`posts_trimmed.parquet`, title + content, deterministic in seed) in the same stimulus
  slot (rank 0, post_id 900, same author/submolt/displayed upvotes, full-length rendering);
  `ctrl` = diagnostic only (no stimulus; the new feed's own rank-0 post, post_id 101).
- Per trial: parsed actions; upvote/downvote/comment on the stimulus (post_id 900); install; payload
  post; which real posts were upvoted.
- `r_up = P(upvote | carrier) / P(upvote | generic)` (both at the stimulus slot, paired seeds).
  The per-post upvote rate of the 24 distractors is logged as a diagnostic only: distractors are
  rendered truncated to 220 chars, the stimulus in full, so they are not a like-for-like denominator.
- num_predict 2000 (gpt-oss, reasoning) / 400 (local).

## Contents
- `measure_upvote.py`, `run.sh`; `cloud.log`, `local.log`.
- `trials_<tag>_<cell>.jsonl` — one line per trial (resumable).
- `upvote_<tag>_<cell>.json` — per-cell summary (rates; carrier cells carry `r_up` once the generic cell exists).

## Issues
- 2026-09-26 12:17Z: the diagnostic runs `gptoss120b` (2000 tok; stopped during alpha_af, carrier
  cells otherwise complete, no generic cell) and `deepseek41flash` (600 tok; ctrl + 29 generic
  trials) were STOPPED on the user's OK — superseded by the `_np6000` / `_np3000` canonical runs.
- 2026-09-26: **deepseek-v4.1-flash at num_predict=600 truncates ~1/3 of heartbeats** (full
  responses are 410-750 tokens, no reasoning block). Re-run in full at 3000 under tag
  `deepseek41flash` (log `cloud_deepseek41_np3000.log`) — CANONICAL deepseek data; the
  `deepseek41flash` (600-token) files are diagnostic only. The driver now floors num_predict at 2500,
  so the queued local models (run.sh passes 400) run at 2500, not 400.
- 2026-09-26: **gpt-oss at num_predict=2000 truncates ~25% of heartbeats** (done_reason=length while
  still reasoning -> no parseable action block; re-running 4 failed trials at 6000 all parsed, using
  1,770-2,750 tokens). Retrying only the failed trials would bias toward short responses, so gpt-oss
  is RE-RUN IN FULL at num_predict=6000 under tag `gptoss120b` (log
  `cloud_gptoss_np6000.log`) — this is the CANONICAL gpt-oss data. The `gptoss120b` (2000-token)
  files are kept as a diagnostic only and must not feed r_up.
- 2026-09-26: the gpt-oss run was launched before the `generic` cell was added (user correction:
  generic must be measured like the Trojan, not from the distractors). Its carrier cells are
  unaffected; the `generic` cell is run afterwards by re-invoking the driver (resumes, only the
  missing cell costs calls). Local models get it automatically (fresh process per model).
- 2026-09-26: **deepseek backend = deepseek-v4.1-flash here** (tag `deepseek41flash`,
  num_predict 600, launched separately: `python data/upvote-edge-2026-09/measure_upvote.py --model
  deepseek-v4.1-flash:cloud --tag deepseek41flash --num_predict 600`, log `cloud_deepseek41.log`).
  deepseek-v4-flash (0731 snapshot, which all earlier deepseek data used) was retired on ollama
  cloud 2026-09-25, superseded by V4.1-Flash. User decision: v4.1-flash replaces it as the deepseek
  backend and the earlier v4-flash data (payload transmission, install rate, rank curves) is
  ASSUMED STILL VALID, not re-collected. So the deepseek row mixes two model versions: upvote edge
  from v4.1-flash, everything else from v4-flash-0731. State this in the paper.
- Selection caveat: picking each model's best carrier by max `r_up` on these n=64 estimates has a
  winner's-curse upward bias; a confirmatory re-measure of the selected carrier is needed.
