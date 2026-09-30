# multihop-rup-2026-09

**Collected:** 2026-09-29
**Collected by:** Claude on birk's behalf (cloud + local ollama)
**Source:** `data/multihop-rup-2026-09/collect.py` (simulated assay; ollama chat only)

## Intent
Measure how the memetic trojan's viral properties evolve under **continued rephrasing**: does
`p(retransmit | post)` and the upvote edge `r_up` change as the same trojan is reworded hop after hop?
Complements the single-shot upvote edge in `data/upvote-edge-2026-09` (depth 0 only) and the multihop
figure `figures/fig_multihop_retransmit.pdf`.

## Method
One run = one model, meme `x_sec_mg`. **Single rephrasing CHAIN with constant-N exposure per hop**
(NOT a closed-loop population; see Issues for two discarded designs). At rephrasing depth `h` the seed
is the `h`-times-reworded trojan (depth 0 = the pristine trojan). Each depth:
- **p(retransmit | post):** a FIXED `n_author=30` agents are exposed to that single seed at rank 0 and
  prefill-forced to author (P(post) assumed fixed). Records the fraction whose post carries the link
  (`p_retransmit_link`) or stays on the carrier topic (bge semantic; `p_retransmit_carrier`).
- **r_up of the seed (trojan arm):** `n_judge=48` agents see the depth-`h` seed at rank 0 on a natural
  heartbeat; `r_up_seed = P(upvote seed)/P(upvote generic)` (the `data/upvote-edge-2026-09` protocol).
- **r_up of a carrier-only post (carrier arm):** the same protocol applied to one payload-*shed* but
  on-carrier-topic response from the same authoring pass (`_pick_carrier_only`, deterministic per depth) ->
  `r_up_carrier`. This is the carrier counterpart of the trojan seed at the same rephrasing depth, so
  facet (b) of `fig_multihop_retransmit` can compare the trojan's upvote pull against the bare carrier's.
  Cloud runs (gpt-oss, deepseek) were collected before this arm existed and were backfilled from their
  saved `responses[]` via `backfill_carrier_rup.py` (judging only; no re-authoring).
- The next seed = ONE randomly chosen link-carrying response (the depth-`(h+1)` rephrasing). If no
  response carries the link, the chain ends. N is fixed every hop, so nothing goes extinct from a
  shrinking exposed population; only the per-post viral rate can change.
- `hops=4`. `num_predict` judging 3500 (cloud) / 2500 (local); author 120. Backends: gpt-oss-120B,
  deepseek-v4.1-flash (cloud); qwen-32B, gemma2-27B, command-r-35B (local, serial, GPU unload).
- Run: `bash run.sh` (both), or `run.sh cloud` and `run.sh local` in parallel (different hardware).

## Replication across chains (for the depth-trend test)
Each depth in a single chain has ONE seed, so the `n_author`/`n_judge` trials at that depth are
pseudo-replicates of one rephrasing -- the noise that matters for "does virality change with depth?" is
BETWEEN chains, not within. `collect.py --chain <k>` runs an INDEPENDENT lineage (same pristine hop-0
trojan, but decorrelated next-seed picks, persona samples and feed seeds) written to
`rup_<tag>_x_sec_mg_c<k>.json`. `run_cloud_chains.sh [N]` runs N extra chains for both cloud backends
concurrently (cloud-only; does not touch the local GPU). The chain (file) is the statistical unit in
`sandbox/multihop_depth_trend.py`, which reports per-chain slopes and their cross-chain consistency
(sign test + t-test). Sign-test power floor is 0.5**K, so K matters: 2 backends alone can never beat
p=0.25; +7 chains/backend -> K=8 (per-backend floor 0.031, pooled ~1.5e-5).

## Contents
- `collect.py` (the assay; `--chain` for extra lineages), `run.sh` (backends: cloud / local targets),
  `run_cloud_chains.sh` (extra cloud chains), `backfill_carrier_rup.py` (carrier arm for old files).
- `rup_<tag>_x_sec_mg.json` per run: `generic_trials[]` and `hops[]`, each depth with
  `p_retransmit_link, p_retransmit_carrier, r_up_seed, p_up_seed, p_upvote_generic, seed` (raw depth-`h`
  seed text), `seed_depth`, `responses[]` (raw authored posts), and `judge_seed[]` (raw per-trial judge
  records: rng seed, parsed, up, actions). Judge feeds are reproducible from each trial's saved seed
  (deterministic `sandbox.real_feed.sample_feed`).

## Issues
- None known so far. With `n_author=30` and p(retransmit) ~0.4, ~12 link responses/hop, so the chain
  sustains; an early chain end (no link response) is flagged via `chain_end`.
- Superseded designs (discarded, files deleted): (1) an arm-based version (separate child/parent runs)
  that conflated payload-shed posts with the Trojan; (2) a closed-loop *population* cascade (each hop's
  pool = all previous-hop posts) where the payload went extinct from a shrinking exposed population
  rather than from any change in its viral properties. This single-chain constant-N design isolates the
  per-post viral rate, which is the quantity of interest.
- Distinct from `sandbox/closed_loop.py` / `data/closed-loop-2026-09` (the free-choice multi-agent
  forum sim) -- unrelated experiment.
