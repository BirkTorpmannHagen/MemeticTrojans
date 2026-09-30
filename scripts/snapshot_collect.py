"""Read-only Moltbook feed snapshotter — collects the OBSERVED visibility/rank
trajectories that the historical single scrape lacks (Workstream 2c).

The HF corpus (data_raw/) is one scrape with only FINAL upvotes and no rank, so
analysis/endogenous_hawkes.py must MODEL visibility from the nominal ranking H and
cannot test upvote-mediation directly. This script polls the live public API on a
schedule and records, per post per poll: its feed RANK (position in the returned
sort), score, hot_score, upvotes/downvotes, comment_count and time. Repeated polls
reconstruct each post's real visibility trajectory -> lets 2a's modeled w_j(t) be
replaced by observed feed rank/score, and validates the nominal H against the
production feed (which independent researchers report can deviate).

SAFETY / SCOPE: strictly READ-ONLY. Only HTTP GET to the public
https://moltbook.com/api/v1/posts endpoint. It NEVER posts, comments, votes,
authenticates, or calls any action endpoint — consistent with the project's
"no real Moltbook actions" rule (which forbids writes, not read-only collection).
Polite: sequential requests, a delay between pages, a descriptive User-Agent, a
bounded page count, and graceful failure on transient errors.

One invocation = one snapshot across the requested sorts, written to
data_snapshots/mb_snapshot_<unixts>.parquet. Schedule it (cron / `claude schedule`
/ a loop) every ~15-30 min to build trajectories. `load_snapshots()` concatenates
all snapshot files for analysis.

    python scripts/snapshot_collect.py                       # one snapshot, all sorts
    python scripts/snapshot_collect.py --sorts hot rising --pages 3
    python scripts/snapshot_collect.py --list                # summarize collected snapshots
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import pandas as pd

API = "https://moltbook.com/api/v1/posts"
OUT_DIR = "data_snapshots"
UA = "moltbook-contagion-research/1.0 (read-only feed snapshotter; visibility trajectories)"
SORTS = ["hot", "new", "rising", "top"]
PAGE_LIMIT = 100          # posts per request (API max observed)
REQUEST_DELAY_S = 0.6     # politeness delay between requests
TIMEOUT_S = 25


def _get(url: str, retries: int = 3):
    """Read-only GET returning parsed JSON, with a couple of polite retries."""
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001 — transient network errors shouldn't abort a poll
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} tries: {url} :: {last}")


def _flatten(post: dict, sort: str, rank: int, snapshot_ts: str) -> dict:
    author = post.get("author") or {}
    submolt = post.get("submolt") or {}
    return {
        "snapshot_ts": snapshot_ts,
        "sort": sort,
        "rank": rank,                                  # 0-based feed position under this sort
        "post_id": post.get("id"),
        "author_id": post.get("author_id"),
        "author_name": author.get("name"),
        "author_karma": author.get("karma"),
        "author_followers": author.get("followerCount"),
        "submolt_name": submolt.get("name"),
        "upvotes": post.get("upvotes"),
        "downvotes": post.get("downvotes"),
        "score": post.get("score"),
        "hot_score": post.get("hot_score"),
        "comment_count": post.get("comment_count"),
        "verification_status": post.get("verification_status"),
        "is_pinned": post.get("is_pinned"),
        "created_at": post.get("created_at"),
    }


def collect_sort(sort: str, snapshot_ts: str, max_pages: int) -> list[dict]:
    rows, cursor, rank = [], None, 0
    for _ in range(max_pages):
        url = f"{API}?sort={sort}&limit={PAGE_LIMIT}"
        if cursor:
            url += f"&cursor={urllib.parse.quote(cursor)}"
        data = _get(url)
        posts = data.get("posts") or []
        if not posts:
            break
        for p in posts:
            rows.append(_flatten(p, sort, rank, snapshot_ts))
            rank += 1
        if not data.get("has_more"):
            break
        cursor = data.get("next_cursor")
        if not cursor:
            break
        time.sleep(REQUEST_DELAY_S)
    return rows


def snapshot(sorts=SORTS, max_pages=3, out_dir=OUT_DIR) -> str:
    os.makedirs(out_dir, exist_ok=True)
    snapshot_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    unix = int(time.time())
    all_rows = []
    for s in sorts:
        rows = collect_sort(s, snapshot_ts, max_pages)
        print(f"  sort={s:7s} {len(rows):4d} posts", flush=True)
        all_rows.extend(rows)
        time.sleep(REQUEST_DELAY_S)
    if not all_rows:
        raise SystemExit("no posts collected — API returned empty (network? rate limit?)")
    df = pd.DataFrame(all_rows)
    path = os.path.join(out_dir, f"mb_snapshot_{unix}.parquet")
    df.to_parquet(path, index=False)
    print(f"snapshot {snapshot_ts}: {len(df)} rows ({df.post_id.nunique()} distinct posts) -> {path}")
    return path


def load_snapshots(out_dir=OUT_DIR) -> pd.DataFrame:
    """Concatenate every collected snapshot into one long frame for trajectory
    analysis (per post_id x sort over snapshot_ts)."""
    files = sorted(glob.glob(os.path.join(out_dir, "mb_snapshot_*.parquet")))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"], utc=True)
    return df


def summarize(out_dir=OUT_DIR):
    df = load_snapshots(out_dir)
    if df.empty:
        print(f"no snapshots in {out_dir}/ yet.")
        return
    n_snaps = df["snapshot_ts"].nunique()
    span = df["snapshot_ts"].max() - df["snapshot_ts"].min()
    print(f"{n_snaps} snapshots over {span} | {df.post_id.nunique()} distinct posts | {len(df)} rows")
    # posts seen in >=2 snapshots = have a trajectory
    per_post = df.groupby(["post_id", "sort"])["snapshot_ts"].nunique()
    traj = int((per_post >= 2).sum())
    print(f"post x sort series with >=2 observations (usable trajectories): {traj}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sorts", nargs="+", default=SORTS, choices=SORTS)
    ap.add_argument("--pages", type=int, default=3, help="max pages per sort (100 posts/page)")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--list", action="store_true", help="summarize collected snapshots and exit")
    args = ap.parse_args()
    if args.list:
        summarize(args.out_dir)
    else:
        snapshot(sorts=args.sorts, max_pages=args.pages, out_dir=args.out_dir)
