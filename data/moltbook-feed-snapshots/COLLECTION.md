# Moltbook live feed snapshots (per-surface visibility)

**Collected:** 2026-08-28 → 2026-08-30 (135 polls, ~44 h); analyzed 2026-09-16.
**Collected by:** the `scripts/moltbook_snapshotter.py` poller (birk's desktop).
**Source:** public Moltbook read-only API, `sort={hot,new,rising,top}`, page limit 100, ~20-min
interval. Raw DB shipped as `moltbook_snapshots.export.db.gz` at the repo root (55 MB gz).

## Intent
The corpus (`data_raw/`, 2026-01-27→02-08) is a single scrape with **no feed rank / no snapshots**,
so the contagion analysis had to *model* post visibility via a nominal ranking `H`. These snapshots
record the **real feed structure over time** for all four ranking surfaces, so feed entry and
persistence — which is what makes or breaks a payload attack — can be measured, not assumed.

## Method
`moltbook_snapshotter.py` polls each sort every ~20 min and appends one row per (poll, sort, rank,
post) to SQLite `snapshots` (cols: snapshot_ts, sort, rank, post_id, author_*, upvotes, downvotes,
score, comment_count, created_at, title, content, ...). Analysis: `analysis/feed_surface_mechanics.py`
(read-only) computes per-surface entry rate, occupant age/upvotes, dwell (persistence), the new-feed
payload-saturation curve, and the list of posts that reached the hot (broadcast) feed.

## Contents
- `moltbook_snapshots.export.db.gz` (repo root) — the raw SQLite, 161,066 rows, 135 polls, 10,602
  distinct posts, 4 sorts.
- Analysis + findings: `analysis/feed_surface_mechanics.py` (`python -m analysis.feed_surface_mechanics`).
- Broadcast-entry probability model: `analysis/hot_entry_probability.py` — P(reach hot/top) from the
  empirical upvote distribution vs entry bar, as a function of a post's upvote-propensity multiplier
  and posting frequency (`out/attack_reach/hot_entry_probability.csv`,
  `docs/figures/hot_entry_probability.pdf`). Validated against the measured entry rates.

## Key findings (2026-09-16)
- **hot = broadcast, karma-gated and rare.** Top-25 occupants are median **31.5 h old, 128 upvotes**;
  entry needs **~111 upvotes over ~23 h**; only **0.58 % (61/10,602)** of posts ever reach it; once in,
  dwell **~16 h** (p90 36 h). The posts that reach it are high-effort organic insights, not spam.
- **new = fresh, fast, SAR-saturable.** 31.6 % of posts pass through; occupants ~0.1 h old, ~4 upvotes;
  dwell only **~0.3 h** (18 min). Recency-ranked, so a payload cascade's share of `new` = `C/(C+557)`
  (ambient ~557 posts/cycle) — high SAR can saturate `new` though it can never reach `hot`.
- **rising / top are static high-karma leaderboards** in this window (occupants ~200 days old, 970 /
  1680 upvotes, dwell = whole window) — not dynamic "rising" surfaces here; treat as non-mechanistic.
- **The nominal `H` (recency + 0.2·upvotes, half-life 3 h) is empirically wrong for `hot`:** the real
  broadcast feed is karma-dominated and persists posts ~1–1.5 days, so fresh posts are essentially
  never in it (0 % of hot occupants are <1 h old). Feed *entry* and *persistence* are the real levers.

## Issues
- Snapshots are 2026-08-28→30, **~7 months after the corpus window** (2026-01/02). Feed *structure* is
  current, not contemporaneous with the corpus contagion estimates — a caveat when combining them.
- `rising`/`top` behaved as static leaderboards during this 44 h window (near-zero genuine throughput);
  do not read them as fresh-post surfaces.
- Audience *size* per surface (how many agents browse each) is NOT in the data — only feed structure.
