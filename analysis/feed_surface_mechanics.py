"""Per-ranking-surface feed mechanics from the live Moltbook snapshots.

Moltbook exposes four feed surfaces (sort = new | rising | hot | top), each with different
entry and persistence dynamics — and entry + persistence is what makes or breaks a payload
attack. This measures, per surface, from `moltbook_snapshots.export.db.gz` (135 polls over
~44 h, 2026-08-28→30, all four sorts, 10,602 distinct posts):

  * ENTRY   — fraction of all seen posts that ever reach that surface's top-25, and the
              upvotes / age needed to enter.
  * PERSIST — dwell time: how long a post stays in the top-25 once there.
  * OCCUPANT profile — age and upvotes of posts holding top-25 slots.
  * NEW-feed turnover -> the payload-saturation calc: `new` is recency-ranked, so a payload
    cascade emitting C posts/cycle competes with the ambient ~557 posts/cycle; the payload's
    share of `new` = C / (C + ambient). High SAR can saturate `new` even though it can never
    reach `hot`.
  * HOT examples — the actual posts that reached the broadcast feed (what goes viral there).

    python -m analysis.feed_surface_mechanics

Read-only analysis of an existing snapshot DB; see data/moltbook-feed-snapshots/COLLECTION.md.
"""
from __future__ import annotations

import datetime as dt
import gzip
import os
import shutil
import sqlite3
import tempfile

import numpy as np

DB_GZ = "moltbook_snapshots.export.db.gz"
SORTS = ["new", "rising", "hot", "top"]
AMBIENT_POSTS_PER_CYCLE = 557     # corpus posts / 30-min cycle (analysis/load)
POLL_MIN = 19.5                   # ~44 h / 135 polls


def _open():
    tmp = tempfile.mktemp(suffix=".db")
    with gzip.open(DB_GZ, "rb") as f, open(tmp, "wb") as o:
        shutil.copyfileobj(f, o)
    return sqlite3.connect(tmp), tmp


def _parse(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def run():
    c, tmp = _open()
    cur = c.cursor()
    total_posts = cur.execute("select count(distinct post_id) from snapshots").fetchone()[0]
    npolls = cur.execute("select count(distinct snapshot_ts) from snapshots").fetchone()[0]
    print(f"snapshot DB: {npolls} polls, {total_posts} distinct posts, sorts={SORTS}\n")
    print(f"{'surface':8}{'entry%':>8}{'occ_age_h':>11}{'occ_up_med':>11}{'enter_up':>10}"
          f"{'dwell_h_med':>12}{'dwell_h_p90':>12}")

    for sort in SORTS:
        rows = cur.execute(
            "select post_id,snapshot_ts,rank,upvotes,created_at from snapshots "
            "where sort=? order by post_id,snapshot_ts", (sort,)).fetchall()
        occ_age, occ_up, enter_up, dwell_h = [], [], [], []
        by_post = {}
        for pid, ts, rk, up, ca in rows:
            by_post.setdefault(pid, []).append((ts, rk, up, ca))
        entered = 0
        for pid, tr in by_post.items():
            in25 = [(ts, rk, up, ca) for ts, rk, up, ca in tr if rk is not None and rk < 25]
            if not in25:
                continue
            entered += 1
            for ts, rk, up, ca in in25:
                try:
                    occ_age.append((_parse(ts) - _parse(ca)).total_seconds() / 3600)
                except Exception:
                    pass
                occ_up.append(up or 0)
            enter_up.append(in25[0][2] or 0)
            # dwell = span of consecutive polls in top-25 (hours), summed
            polls_in = len(in25)
            dwell_h.append(polls_in * POLL_MIN / 60.0)
        entry_pct = 100 * entered / total_posts
        print(f"{sort:8}{entry_pct:>8.2f}{np.median(occ_age):>11.1f}{np.median(occ_up):>11.0f}"
              f"{np.median(enter_up):>10.0f}{np.median(dwell_h):>12.1f}{np.percentile(dwell_h,90):>12.1f}")

    # --- NEW-feed payload saturation vs cascade emission rate ---
    print("\nNEW-feed payload saturation (recency-ranked; payload share = C/(C+ambient), "
          f"ambient={AMBIENT_POSTS_PER_CYCLE}/cycle):")
    for C in [10, 50, 100, 300, 557, 1000, 3000]:
        print(f"  cascade emits {C:5d} payload posts/cycle -> {100*C/(C+AMBIENT_POSTS_PER_CYCLE):5.1f}% of the new feed")

    # --- HOT examples: what actually reached the broadcast feed ---
    print("\nHOT (broadcast) feed — examples of posts that reached top-25 (peak upvotes):")
    ex = cur.execute(
        "select post_id, max(upvotes) mu, min(rank) mr, title from snapshots "
        "where sort='hot' and rank<25 group by post_id order by mu desc limit 12").fetchall()
    for pid, mu, mr, title in ex:
        t = (title or "")[:70].replace("\n", " ")
        print(f"  up={mu:>4}  best_rank={mr:>2}  {t}")
    os.remove(tmp)


if __name__ == "__main__":
    run()
