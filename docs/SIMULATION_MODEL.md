# Reach / install simulation — model, assumptions, and code audit

> **Simulation model v2 — surface×cascade, pinned P(post)** (2026-09-24). Current entry point:
> `analysis.make_paper_figures` → `sandbox.expected_installs_surface` (+ `expected_installs_edge`);
> per-model SAR / curve / install-rate helpers come from `sandbox.reach_by_model` and
> `sandbox.ccdf_trojan_baseline`. Retransmission SAR is pinned:
> `SAR = P(post)=0.10 · P(payload|post, security)`, so all five backends are assessable and the
> per-model signal is the prefill-measured payload transmission (the earlier dedicated-sweep
> "measured" SAR of 12.5/22.5/178‰ is retired). This doc is kept honest by
> `python -m docs.audit_simulation_model`.

This document is the single source of truth for the **reach / payload-install
simulation** (the third leg of the study: observed contagion → per-agent assay →
projected reach). It records every assumption and mechanism, and audits each one
against the code that implements it. If code and doc disagree, that is a bug in one
of them — the audit table at the end is meant to catch exactly that.

Scope: the **state-mediated feed cascade** engine and the generators that drive it.
It does **not** cover meme mining or the observational contagion estimates.

## 1. One engine, one config

All reach/install generators call a single engine, `sandbox.endogenous_sim.simulate_endo`,
parameterised by one shared frozen config, `sandbox.endogenous_sim.CONFIG` (a `SimConfig`).
Generators differ **only** in the retransmission input (which encodes the scenario) and
in bookkeeping — never in the exposure model, population size, horizon, or feed geometry.

| Generator | Output | Retransmission input (scenario) |
|---|---|---|
| `sandbox.expected_installs_surface` (+`_edge`) | Tables 1/A/B + `expinst_ccdf.pdf` (state + edge panels) | **Pinned population** — `p_ret = 0.10 · P(payload\|post)` per model; carrier enters via the upvote edge `r_up` |

The state-mediated table/CCDF come from `expected_installs_surface`; the edge panel from
`expected_installs_edge` (Twitter follower graph). Pure per-model inputs (the pinned
`SAR_MODELS = 0.10 · P(payload|post)`, feed-position curves, install rates) are provided as
data helpers by `sandbox.reach_by_model` and `sandbox.ccdf_trojan_baseline` — they build no
cascade of their own. The phase-diagram sweep `sandbox.endogenous_sim.run` uses the same engine,
sweeping `p_ret` (SAR) against the realized view→upvote rate.

## 2. Exposure is counted in VIEWS

A post at feed rank `r` is **seen by `spread_any(r) · N` agents** that heartbeat, where
`spread_any(r)` is the assay-measured probability a feed-checking agent sees a post at
rank `r`, and `N` is the whole population. This is the state-mediated broadcast: a post at
the **top** of the global karma feed reaches ~all of `N` (`spread_any(0) ≈ 0.94`).

Exposure is **not** counted in upvotes or comments. (The previous engine used
`engage0 · view(rank)` with `engage0 = 6` calibrated to the corpus *upvote* scale — an
undercount of viewers by ~1–2 orders of magnitude, and inconsistent with the
`P(install | seen)` install rate. That is removed.)

Distinct reach depletes a shared susceptible pool of size `N` (no agent is counted twice). The
feed is a **shared global surface**, so an agent sees the meme at its **best-placed** live post:
per cycle `new = remaining_sus · max(visᵢ)`. It is **not** `1 − Πᵢ(1 − visᵢ)` — that would treat
hundreds of buried retransmitted duplicates as independent exposure chances and compound to fake
network saturation (the bug behind "one top-25 post dooms everyone"; fixed 2026-09-16).

## 3. Earned climb (how a fresh post enters the feed)

A freshly seeded, 0-upvote payload post does **not** start at the top. It is buried under
the corpus-karma competitor field (see §6) and its organic rank is deep, so `spread_any(rank) ≈ 0`.
Its only guaranteed visibility is a **cold-start foothold**: a fresh post (age 0) is floored at
the **bottom visible feed slot**, `coldstart_rank = K_FEED − 1 = 24`, i.e. it is seen by
`spread_any(24) · N ≈ 0.00282 · 37189 ≈ 105` agents — the "new/rising" surface.

From that foothold a post reaches the whole network **only** if it is carried to the top slots by
**retransmission** (fresh posts re-enter the feed at high recency) and, in principle, upvote
amplification (§5). This makes carrier reach contagion-dependent and bimodal (most seeds fizzle
near the cold-start floor; amplified ones saturate) — rather than every seed trivially reaching `N`.

**This cold-start audience (`spread_any(K−1)·N ≈ 105`) is the load-bearing knob.** Criticality of
the branching cascade is roughly `R ≈ 105 · SAR`: SAR=0 fizzles at ~105; at the pinned per-model
SAR (≈25–50‰, see below) the cascade is supercritical and the reach distribution is bimodal (most
seeds fizzle near the cold-start floor, amplified ones climb toward `N`). `spread_any(24)` is an
exponential extrapolation from assay ranks {0, 5} (§7), so this value carries real uncertainty; the
whole reach ladder shifts with it. It is stated explicitly here so it can be challenged.

## 4. Retransmission = SAR = P(post) · P(payload | post) — NEW POST only, via prefill

**Broadcast retransmission = a NEW payload-carrying post only.** Only a new post re-enters the
global feed and reaches new agents; a **comment** sits under the original post and does **not**
broadcast, so **comment retransmission is out of scope** and is never a sim input (project decision;
memory `retransmission-newpost-prefill`).

Each agent that sees a payload post **retransmits** — authors a fresh payload-carrying post — with
probability `p_ret = SAR = P(post) · P(payload | post)`, measured by the **prefill decomposition**
(`security_warning` carrier): `P(post)` sampled naturally,
`P(payload | post)` forced by prefilling a `create_post` start. `P(post)` is then pinned at 0.10, so
`SAR = 0.10 · P(payload|post)` per model (`SAR_MODELS` in `reach_by_model`; values below). Children are
new age-0 payload posts that re-enter buried and must climb again.

**Why prefill, not the natural `payload_post`:** the 2026-09 cloud rank-curve collection
(`data/rank-curves-cloud-2026-09`) shows the *natural*
new-post retransmission rate `payload_post ≈ 0` for both gpt-oss (0.000 at every rank) and deepseek
(0–0.013) — a ~1% rate is 0–1 events at n=80, too rare to measure at feasible n, and both models
re-transmit almost entirely via comments (out of scope). So natural `payload_post` is **not** used;
the prefill-coupled `P(post)·P(payload|post)` is the estimator instead. Caveat: prefill
`P(payload|post)` (forced) can exceed the natural free-choice rate, so this SAR is an **upper
estimate** of new-post retransmission. The unified assay's `payload_post` / `payload_tx` curves are
kept as **diagnostics** (they reveal the comment-only sink pattern), never as the cascade input.

- **`reach_by_model` (heterogeneous population):** each viewer is drawn from the model mix and
  retransmits at that model's SAR. `P(post)` is pinned at `0.10` (a shared volume constant), so
  `SAR_MODELS = 0.10 · P(payload|post, security)` (‰): gpt-oss-120B 50, qwen-32B 37.5,
  deepseek-v4-flash 46.9, plus an equal-share mixture. (Previously used the dedicated-sweep measured
  SAR — 12.5/22.5/178 — which baked in inconsistent per-model `P(post)`; repinned 2026-09 for
  consistency.)
- **`expected_installs_surface` (+`_edge`; the state/edge installs tables):** `P(post)` is pinned
  at a common `0.10`, so `p_ret = 0.10 · P(payload|post)` per model and the carrier enters only via
  the upvote edge `r_up`. All five backends are assessable (no dedicated SAR sweep needed).

Both are `P(post) · P(payload|post)`; they differ only in whether `P(post)` is per-model-measured
(heterogeneous) or pinned (homogeneous).

Measured per-model `P(post)` (for reference; `out/exposure/{sampled_sar,ollama_sar}_*.json`):
gpt-oss-120B 0.025, qwen-32B 0.030, qwen-3B 0.053, qwen-14B 0.042, deepseek-v4-flash 0.375.

## 5. Amplification (upvote climb) — present but negligible

Agents that see a post **upvote** it with the realized view→upvote rate `cfg.view_upvote = 0.02`,
raising its karma (`score = recency + 0.2 · upvotes`) so it can climb. This is **not** the assay's
`P(upvote | seen) = 0.73`: that is a forced-attention single-agent rate, and applying it to
viewer-scale exposure would make climbing trivial (or, at corpus-scale competitors, instantly pin the
meme at rank 0). At any realistic view→upvote rate (0–10%) reach is **insensitive** to it — the
cascade is dominated by cold-start × retransmission, so the endogenous upvote loop is second-order.
The rate is kept as a config knob so the phase diagram can show where it *starts* to matter (well
above realistic values).

**How much more is the Trojan upvoted than a normal post?** (informs the climb advantage.)
Measured, not assumed:
- **Real corpus** (posts, `analysis.load`): security/verification-themed meme-bearing posts accrue
  **1.2–1.35×** the upvotes of the platform baseline (`security` word 1.35×, `github.com` 1.28×,
  `how do you handle` 1.32×, `guard*` 1.24×, 🛡 1.18×; neutral phrases ≈ 1.0×). Baseline mean ≈ 2.2.
- **Assay** (`unified_openai_*.json`, rank 0): the Trojan carrier `security_warning` has
  `P(upvote|seen)=0.857` vs a control persona post `0.804` — a **1.07×** advantage.

So the Trojan's amplification edge over a normal post is **modest** (~1.1–1.35×), not order-of-magnitude.
Given the insensitivity above, this modest edge does not change the qualitative reach result; it is
recorded so the climb is grounded in a measurement rather than an assumption.

## 5a. Rank-conditional reach/installs — the climb-free alternative (`sandbox/rank_conditional.py`)

Because reach is so rank-dependent and any climb model carries assumptions, the primary robust
statement **conditions on the rank the seed achieves** instead of modelling how it gets there
(engine `pin_rank` mode: every payload post is held at feed rank `r`, so views = `spread_any(r)·N`
per cycle, no cold-start/competitor/upvote machinery). Sweeping `r`:

- **Direct (single seed, no retransmission)** — the assumption-light lower bracket. Expected installs
  fall smoothly with rank: **rank 0 ≈ 31,400 → rank 5 ≈ 31,200 → rank 11 ≈ 20,000 → rank 17 ≈ 6,500
  → rank 24 (bottom slot) ≈ 1,300** (installs = reach · mean install rate ≈ 0.84).
- **With per-model retransmission** (payload held at `r`): the cascade is **supercritical from every
  rank** — even the bottom slot — for all measured SARs (all exceed the criticality threshold
  `1/(spread_any(24)·N)·(1/lifetime)`), so it saturates to `N · install_rate` regardless of `r`.

Reading: *if the payload merely appears anywhere in the feed and retransmits at the measured SAR,
containment fails; if retransmission is blocked, the single-post install count is set by the rank it
reaches* (the direct curve). Output: `docs/rank_conditional_table.tex`,
`figures/rank_conditional.pdf`, `out/attack_reach/rank_conditional.csv`.

## 5b. Per-surface feed model — the empirical reframe (SUPERSEDES the nominal-H cascade)

Live Moltbook snapshots (`data/moltbook-feed-snapshots/`, `analysis/feed_surface_mechanics.py`;
135 polls, 4 sorts) show the reach question is really **feed entry × persistence, per ranking
surface** — and the nominal `H` (recency + 0.2·upvotes, half-life 3 h) that the cascade engine
assumes is **empirically wrong** for the broadcast feed.

| surface | entry rate | occupant age / upvotes | dwell | mechanics for an attack |
|---|---|---|---|---|
| **new** | 31.6% of posts | 0.1 h / ~4 up | ~0.3 h (18 min) | recency-ranked → payload share = `C/(C+557)` (ambient posts/cycle); **high SAR can saturate `new`**, but per-post persistence is minutes |
| **hot** (broadcast) | **0.58%** | 31.5 h / 128 up | ~16 h (p90 36 h) | karma-gated: entry needs **~111 upvotes over ~23 h**; rare but enormous & persistent; posts that reach it are high-effort organic insights, not spam |
| rising / top | 0.24% | ~200 d / 970–1680 up | whole window | static high-karma leaderboards here; non-mechanistic |

**Consequences for the sim:**
1. A fresh, 0-karma payload post is **essentially never in `hot`** (0% of hot occupants are <1 h old).
   So the earlier "a single top-25 post installs on `install(r)·N`" is a `hot`-feed *conditional* that
   almost never fires for a seeded post; it is not the expected outcome.
2. The `simulate_endo` cascade — no post ever ranks, all reach via cold-start — is really a **`new`-feed
   model** (fast turnover, SAR-saturable), and should be framed and calibrated as such (share of `new`,
   `new`-audience × persistence), not as broadcast.
3. **`hot` broadcast is a separate, rare, karma-gated event**: model P(payload reaches hot) from the
   empirical base rate (~0.6% of posts, ~1.8% of `new`-feed posts), lifted by the payload's ~1.2–1.35×
   upvote edge, with ~16 h of broadcast dwell if it lands. The attack's tail risk lives here.
4. The reach headline is therefore **bimodal by construction**: a steady `new`-feed trickle for most
   seeds, plus a rare `hot`-entry jackpot — *entry and persistence*, not a smooth cascade, decide it.

The `simulate_endo` broadcast/cold-start numbers (the state panel of `expinst_ccdf.pdf`) are a `new`-feed-scale
bound; the per-surface model above is the current understanding. (Caveat: snapshots are ~7 months
after the corpus window — feed *structure* is current, not contemporaneous.)

## 5c. Edge-mediated substrate — the retransmission-focused view (`sandbox/reach_edge.py`)

The state-mediated feed decouples reach from retransmission (broadcast is a rare karma gate). The
**edge-mediated** cascade puts retransmission at the center: the payload spreads
agent→follower→follower on the SNAP **ego-Twitter** follower graph (`data/ego-twitter-snap/`,
N=81,306, mean degree 29.8), and each follower's feed is its neighbours' posts in **random order**
— so there is **no karma/upvote channel** and reach is driven purely by **SAR × graph structure**.

Mechanism (`simulate_edge_feed`, a random-rank neighbour-feed cascade): a candidate
follower `w` has feed size `F = 1 + Poisson(outdeg[w]·post_rate·window)`; the payload sits at a
**uniform-random** rank `r∈[0,F)`; it is seen w.p. `spread_any(r)` (0 past K); of those who see,
install w.p. the per-model rank-0 ASR and **retransmit at a RANK-RESOLVED rate**
`SAR₀·exp(-k·r)` (marginal), where `SAR₀ = P_POST_REF·P(payload|post)` and `k` is the measured
SAR-by-rank decay (`sar_by_rank_gptoss120b`, `k=0.134`, ~25× over the feed; gpt-oss is the only model
collected, used as the fallback decay for the rest). Hubs' large feeds bury the payload
(degree-dependent dilution). Five models have both arms (gpt-oss, deepseek, qwen-32B, gemma2-27B,
command-r-35B; command-r has no measured install → 0.66 fallback). Arms differ in SAR₀ (Trojan
`cross_child` tx vs bare generic `cross_bare` tx).

**Result (1,000 MC cascades, median seed):** amplification (Trojan/generic) is **~1.3–2.1×**
(gpt-oss 2.1, deepseek 1.8, qwen-32B 1.5, gemma2-27B 1.3 — its generic tx is nearly as high as Trojan,
command-r 1.3). **Folding in the measured rank-resolved SAR is load-bearing: it collapses
amplification from ~10–17× (flat-SAR, which this earlier used and which OVERSTATED it) to ~1.3–2.1×** —
because retransmission is concentrated at the top of the feed (`exp(-0.134r)`), but a random-order
neighbour feed usually *buries* the payload, so it rarely gets re-posted. Edge-mediated amplification
thus lands in roughly the same band as the state-mediated per-surface model, not 10× above it. Tails
shrink accordingly (gpt-oss Trojan: mean 15.6 installs, p99 181; generic p99 20). The paper's edge
install CCDF is the edge panel of `figures/expinst_ccdf.pdf`; `reach_edge.run` also emits
`figures/tab_amplification_edge.tex`,
`out/attack_reach/{reach_raw_edge,reach_by_model_edge,amplification_edge}.csv`. Caveats: ego-Twitter
is a structural stand-in (Moltbook exposes no follower graph); the SAR-decay `k` is gpt-oss-measured
and reused for the other models (only gpt-oss's SAR-by-rank was collected).

## 6. Competitor field (feed ranking reference)

To assign a rank to each payload post, the engine draws a **static** field of `n_live = 6685`
competitor posts (mean live posts = corpus posts/cycle × 6 h window), each with
`score = recency(age ~ U[0, WINDOW_H]) + 0.2 · upvotes`, upvotes sampled from the real corpus
distribution (mean ≈ 2.2, median 2, max 228). A payload post's rank = number of competitors
outranking it. The field is static (not co-evolving): because amplification is negligible (§5),
competitor karma dynamics are second-order for reach.

## 7. Inputs and provenance

- **Rank curves** `spread_any(r)`, `install(r)`, `upvote(r)`: `load_params` fits `y0·exp(−k·r)` to
  `out/exposure/unified_openai_gpt-4o-mini.json`, which has only ranks **{0, 5}** — so every curve is
  a 2-point exponential extrapolated across ranks 0…24. `spread_any(0)=0.941`, `spread_any(5)=0.280`,
  `spread_any(24)≈0.0028`. **Caveat:** the rank curves come from gpt-4o-mini, which is otherwise
  quarantined from this study; they are retained only as the exposure geometry (see the memory note
  `gpt4omini-quarantined`). The deep-rank extrapolation (§3) is the most fragile part.
- **Install rate** `P(install | seen)` at rank 0 (per-model ASR): `out/exposure/asr_model_*.json` —
  gpt-oss-120B 1.00, qwen-32B 0.80, deepseek-v4-flash 0.733. Applied to distinct reach:
  `installs = reach · install_rate`.
- **SAR / P(post) / P(payload|post):** `out/exposure/{sampled_sar,ollama_sar}_*.json`
  and the per-model `cross_child_*` prefill files.
- **Recency half-life:** `fitted_half_life()` → `out/exposure/exposure_params.json`
  (`engagement_half_life_h ≈ 20.3 h`; fallback nominal 3 h). `CYCLE_H = 0.5 h`, `WINDOW_H = 6 h`,
  post lifetime `maxage = WINDOW_H/CYCLE_H + 2 = 14` cycles, `K_FEED = 25`.
- **Population:** `N = 37189` (fixed `SimConfig.N`, the Moltbook agent-population size).

## 8. Substrates

Two substrates, one per mediation mechanism. **The Moltbook comment/reply interaction graph is
not used** — Moltbook enters state-mediated, and the only explicit edge graph is Twitter's.

- **State-mediated (primary):** the feed-visibility broadcast (`endogenous_sim.simulate_endo` /
  `simulate_surface_cascade`, driving `expected_installs_surface`). A payload at feed rank `r` is
  seen by a `spread_any(r)` fraction of the bounded-active population; seed-independent. This is
  how Moltbook contagion is modelled.
- **Edge-mediated:** the **SNAP ego-Twitter follower graph** (`reach_edge.simulate_edge_feed`,
  driving `expected_installs_edge`): a payload spreads agent → follower along a real scale-free
  follower topology (`data_ext/twitter_combined.txt.gz`), seeded by follower-count percentile.
  This is an external topology, not derived from Moltbook.

## 9. Config values (`SimConfig`, the shared defaults)

| Field | Value | Meaning |
|---|---|---|
| `N` | 37189 | total population (fixed) |
| `cycles` | 120 | 30-min heartbeat cycles simulated |
| `n_live` | 6685 | competitor live-post field size |
| `coldstart_rank` | 24 (`K_FEED−1`) | fresh post's guaranteed bottom feed slot |
| `view_upvote` | 0.02 | realized fraction of viewers who upvote (amplification); reach insensitive over 0–0.1 |
| `maxchildren` | 400 | per-cycle retransmission cap (numerical stability) |

## 10. Known limitations

1. **Measured rank curves now exist for gpt-oss-120B & deepseek-v4-flash** (9 feed ranks 0–24,
   `data/rank-curves-cloud-2026-09/`, `out/exposure/unified_{gptoss120b,deepseekflash}.json`),
   replacing the 2-point {0,5} gpt-4o-mini extrapolation for those two models. **qwen and any other
   model still fall back to gpt-4o-mini** until collected (`ccdf_trojan_baseline.unified_for`).
2. **Visibility is resolved by the broadcast reading, not a separate curve.** The assay shows the
   agent the whole 25-post feed, so a per-rank **action** curve — e.g. `install(r)` — is already the
   **network action fraction** when the payload holds that feed position (top-K seen by all each
   heartbeat). So a single payload post at rank r installs on `install(r)·N` agents directly — no
   `spread_any×N × install_rate` double-count. Measured: gpt-oss `install(r)=0.89–0.99` (flat →
   33k–37k installs), deepseek `0.60–0.78` (22k–29k), **near rank-independent** — a single top-25
   post is enough for soft-target models (from the rank curves in `data/rank-curves-cloud-2026-09/`).
3. **The multi-hop CASCADE still hangs on an unmeasured per-cycle activity fraction.** Under full
   broadcast, retransmission (SAR·N per post per heartbeat) is explosively supercritical for any
   nonzero SAR — so the cascade's criticality is set by how many agents actually act per cycle, which
   the assay does not measure. The *single-post* installs-by-rank above is the assumption-light,
   robust statement; the branching cascade is not, and should be reported with that caveat (or the
   activity fraction stated explicitly).
4. **Retransmission is rank-0 prefill only.** `P(post)`/`P(payload|post)` were never swept over rank
   (all `sampled_sar_*`/`ollama_sar_*` are position 0); natural `P(post)` by rank is ~1–4% and too
   noisy. So `p_ret` is rank-independent for now. A prefill-by-rank sweep would resolve it.
5. **Static competitor field — competitors do NOT climb.** The meme can accrue upvotes and climb
   against a frozen field, an asymmetry only moot because the meme's own climb is negligible at
   realistic view→upvote rates. If amplification matters, make the competitor field co-evolve (upvote
   climb + turnover, as the older `simulate_global`/`comp_mode="control"` did).
4. **P(post) is pinned, not measured.** `expected_installs` (state + edge) pins `P(post)=0.10` as a
   shared volume constant; the per-model signal is the prefill `P(payload|post)` only.
5. **A separate, older upvoter-scale engine still exists** (`endogenous_sim_conserved`, used by
   `make_install_tables`); it is **not** on this shared config and should be migrated or retired.

## 11. Audit — doc vs. code

Checked against `sandbox/endogenous_sim.py` (`simulate_endo`, `simulate_surface_cascade`, `SimConfig`)
and the current generators (`expected_installs_surface` + `_edge`; per-model data helpers in
`reach_by_model` / `ccdf_trojan_baseline`).

Run `python -m docs.audit_simulation_model` to re-verify; it exits non-zero if any claim
below no longer matches the code. Last run: **all PASS**.

| # | Claim in this doc | Code location | Status |
|---|---|---|---|
| A | Exposure = `spread_any(rank)·N` (views) | `simulate_endo`: `vis = spread_any(min(above,K−1))` | ✅ PASS |
| B | Cold-start floor = `spread_any(K−1)` for age-0 posts (baseline reach = `f_new·N` ≈ 105) | `vis = max(vis, where(age==0, f_new))`, `f_new = curve(...,coldstart_rank)` | ✅ PASS |
| C | Shared-feed reach `new = rem·max(vis)` (best-placed post, NOT OR-compounded over duplicates) | `p_any = float(vis.max()); new = remaining_sus·p_any` | ✅ PASS |
| D | Retransmission = `binomial(new, p_ret)`, `p_ret` = SAR | `children = binomial(n_new, p_ret)` | ✅ PASS |
| E | Installs = reach · install_rate | `cum_inst += new · install_rate` | ✅ PASS |
| F | Amplification uses `cfg.view_upvote`, not assay 0.73; reach insensitive over 0–0.1 | `up += attrib · cfg.view_upvote` | ✅ PASS |
| G | `N=37189`, `cycles=120` shared; state generators default to `CONFIG` | `CONFIG`; `rank_conditional` | ✅ PASS |
| H | *(retired with `expected_installs_table.py`; the pinned-P(post) installs model is now `expected_installs_surface`)* | — | — |
| I | Moltbook comment/reply graph removed (`build_edges`/`build_graph` gone; twitter-only edges) | no `build_edges`/`build_graph` in `attack_reach_sim`/`reach_by_model`/`ccdf_trojan_baseline`/`load` | ✅ PASS |
| J | `pin_rank` mode holds every post at rank r; SAR=0 direct reach = `N·(1−(1−spread_any(r))^maxage)` | `simulate_endo(..., pin_rank=r)` | ✅ PASS |
| K | Surface×cascade engine exists (per-seed broadcast break-in) | `endogenous_sim.simulate_surface_cascade(seed_bcast=…)` | ✅ PASS |
| L | State model pins `P(post)=0.10`; `SAR=0.10·P(payload\|post)`; 5 backends | `expected_installs_surface`: `P_POST_STATE=0.10`, `sar = P_POST_STATE * _p_payload`, `MODELS` (5) | ✅ PASS |
| M | `reach_by_model.SAR_MODELS` pinned = `1000·0.10·P(payload\|post)` (old 12.5/22.5/178 retired) | `P_POST_PIN=0.10`, `SAR_MODELS = round(1000·P_POST_PIN·_ppl(tag))` → 50/37.5/46.9‰ | ✅ PASS |
| N | Cell = installs + generic base + non-tx base + `Amp_gen` + `Amp_ntx` | `expected_installs_surface`: `gbase`(r_up=1, retx on), `nbase`(retx off), `amp`, `amp_nt` | ✅ PASS |
| O | Edge model shares the pinned `P_POST_REF=0.10` | `expected_installs_edge` / `reach_edge.P_POST_REF=0.10` | ✅ PASS |
