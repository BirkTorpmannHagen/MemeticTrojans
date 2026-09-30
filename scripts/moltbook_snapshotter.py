#!/usr/bin/env python3
"""Moltbook feed snapshotter — standalone, dependency-free, self-scheduling.

Collects OBSERVED feed-visibility trajectories from the live Moltbook API so the
contagion analysis can replace its MODELED post visibility with real feed rank/
score over time (Workstream 2c). Runs unattended on a reliable always-on machine
(a desktop), polling every N minutes and appending to a single SQLite file.

WHY A SEPARATE SCRIPT: this needs to run for days/weeks on a machine that does not
sleep. It uses ONLY the Python standard library (no pandas/pyarrow/venv), stores to
one SQLite file that survives restarts, and schedules itself (no cron needed).

SAFETY / SCOPE: strictly READ-ONLY. Only HTTP GET to the public
https://moltbook.com/api/v1/posts endpoint. It NEVER posts, votes, comments,
authenticates, or calls any write/action endpoint.

QUICK START (on the desktop):
    python3 moltbook_snapshotter.py                 # poll every 20 min, forever
    python3 moltbook_snapshotter.py --interval 15 --pages 3
    python3 moltbook_snapshotter.py --once          # single poll, then exit (test)
    python3 moltbook_snapshotter.py --stats         # summarize what's collected
    python3 moltbook_snapshotter.py --export snapshots.csv   # dump to CSV to move

Leave it running with, e.g.:
    nohup python3 moltbook_snapshotter.py --interval 20 > snap.log 2>&1 &   # mac/linux
    (or a screen/tmux session; on Windows, a scheduled task or just a terminal)

Requires: Python 3.8+ only. Data lands in ./moltbook_snapshots.db (override --db).
Stop cleanly with Ctrl-C — the current poll finishes and commits first.
"""

from __future__ import annotations

import argparse
import json
import signal
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

API = "https://moltbook.com/api/v1/posts"
UA = "moltbook-contagion-research/1.0 (read-only feed snapshotter; visibility trajectories)"
SORTS = ["hot", "new", "rising", "top"]
PAGE_LIMIT = 100
REQUEST_DELAY_S = 0.6      # politeness delay between page requests
TIMEOUT_S = 25
CONTENT_CHARS = 6000       # cap stored content length (posts are short; bounds DB size)

_STOP = False


def _on_signal(signum, frame):
    global _STOP
    _STOP = True
    print(f"\n[{_now()}] signal {signum} received — finishing current poll, then exiting.",
          flush=True)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# Storage                                                                      #
# --------------------------------------------------------------------------- #
DDL = """
CREATE TABLE IF NOT EXISTS snapshots (
    poll_id       INTEGER NOT NULL,   -- one value per poll cycle (unix seconds at poll start)
    snapshot_ts   TEXT    NOT NULL,   -- ISO-8601 UTC of the poll
    sort          TEXT    NOT NULL,   -- hot | new | rising | top
    rank          INTEGER NOT NULL,   -- 0-based position in that sort == observed feed visibility
    post_id       TEXT    NOT NULL,
    author_id     TEXT,
    author_name   TEXT,
    author_karma  INTEGER,
    author_followers INTEGER,
    submolt_name  TEXT,
    upvotes       INTEGER,
    downvotes     INTEGER,
    score         INTEGER,
    hot_score     REAL,
    comment_count INTEGER,
    verification_status TEXT,
    is_pinned     INTEGER,
    created_at    TEXT,               -- post creation time (for age = snapshot_ts - created_at)
    title         TEXT,
    content       TEXT,               -- needed to detect which posts carry a meme (Goal B)
    PRIMARY KEY (poll_id, sort, post_id)
);
CREATE INDEX IF NOT EXISTS idx_post_time ON snapshots (post_id, snapshot_ts);
CREATE INDEX IF NOT EXISTS idx_poll ON snapshots (poll_id);
"""


def open_db(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path, timeout=30)
    con.execute("PRAGMA journal_mode=WAL;")     # durable, restart-safe appends
    con.execute("PRAGMA synchronous=NORMAL;")
    con.executescript(DDL)
    con.commit()
    return con


# --------------------------------------------------------------------------- #
# Fetch (read-only)                                                            #
# --------------------------------------------------------------------------- #
def _get(url: str, retries: int = 4):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001 — transient network errors must not kill the loop
            last = e
            time.sleep(2.0 * (attempt + 1))     # simple backoff
    raise RuntimeError(f"GET failed after {retries} tries: {url} :: {last}")


def _row(post: dict, sort: str, rank: int, poll_id: int, ts: str) -> tuple:
    a = post.get("author") or {}
    sm = post.get("submolt") or {}
    content = (post.get("content") or "")[:CONTENT_CHARS]
    return (
        poll_id, ts, sort, rank, post.get("id"),
        post.get("author_id"), a.get("name"), a.get("karma"), a.get("followerCount"),
        sm.get("name"), post.get("upvotes"), post.get("downvotes"), post.get("score"),
        post.get("hot_score"), post.get("comment_count"), post.get("verification_status"),
        1 if post.get("is_pinned") else 0, post.get("created_at"),
        post.get("title"), content,
    )


def collect_sort(sort: str, poll_id: int, ts: str, max_pages: int) -> list[tuple]:
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
            rows.append(_row(p, sort, rank, poll_id, ts))
            rank += 1
        if not data.get("has_more"):
            break
        cursor = data.get("next_cursor")
        if not cursor:
            break
        time.sleep(REQUEST_DELAY_S)
    return rows


INSERT = """INSERT OR IGNORE INTO snapshots
(poll_id, snapshot_ts, sort, rank, post_id, author_id, author_name, author_karma,
 author_followers, submolt_name, upvotes, downvotes, score, hot_score, comment_count,
 verification_status, is_pinned, created_at, title, content)
VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""


def poll_once(con: sqlite3.Connection, sorts, max_pages) -> int:
    poll_id = int(time.time())
    ts = _now()
    total = 0
    for s in sorts:
        try:
            rows = collect_sort(s, poll_id, ts, max_pages)
        except Exception as e:  # noqa: BLE001 — one bad sort shouldn't drop the whole poll
            print(f"[{ts}]   sort={s} FAILED: {e}", flush=True)
            continue
        con.executemany(INSERT, rows)
        total += len(rows)
        print(f"[{ts}]   sort={s:7s} {len(rows):4d} posts", flush=True)
        time.sleep(REQUEST_DELAY_S)
    con.commit()
    return total


# --------------------------------------------------------------------------- #
# Reporting / export                                                           #
# --------------------------------------------------------------------------- #
def stats(con: sqlite3.Connection):
    q = con.execute("SELECT COUNT(*), COUNT(DISTINCT poll_id), COUNT(DISTINCT post_id), "
                    "MIN(snapshot_ts), MAX(snapshot_ts) FROM snapshots").fetchone()
    rows, polls, posts, tmin, tmax = q
    if not rows:
        print("no snapshots collected yet.")
        return
    traj = con.execute(
        "SELECT COUNT(*) FROM (SELECT post_id, sort FROM snapshots "
        "GROUP BY post_id, sort HAVING COUNT(DISTINCT poll_id) >= 2)").fetchone()[0]
    print(f"{rows} rows | {polls} polls | {posts} distinct posts")
    print(f"window: {tmin} -> {tmax}")
    print(f"post x sort series with >=2 observations (usable trajectories): {traj}")


def export_csv(con: sqlite3.Connection, path: str):
    import csv
    cur = con.execute("SELECT * FROM snapshots ORDER BY poll_id, sort, rank")
    cols = [d[0] for d in cur.description]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        n = 0
        for row in cur:
            w.writerow(row)
            n += 1
    print(f"exported {n} rows -> {path}")


# --------------------------------------------------------------------------- #
# Main loop                                                                    #
# --------------------------------------------------------------------------- #
def run(db, sorts, interval_min, max_pages, once, max_hours):
    con = open_db(db)
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    interval_s = interval_min * 60.0
    deadline = time.time() + max_hours * 3600 if max_hours else None
    print(f"[{_now()}] snapshotter start | db={db} | every {interval_min} min | "
          f"sorts={','.join(sorts)} | pages={max_pages}"
          + (f" | stop after {max_hours}h" if max_hours else " | until Ctrl-C"), flush=True)

    poll_n = 0
    while not _STOP:
        t0 = time.time()
        try:
            n = poll_once(con, sorts, max_pages)
            poll_n += 1
            print(f"[{_now()}] poll #{poll_n}: {n} rows committed.", flush=True)
        except Exception as e:  # noqa: BLE001 — keep the collector alive no matter what
            print(f"[{_now()}] poll error (continuing): {e}", flush=True)
        if once or _STOP:
            break
        if deadline and time.time() >= deadline:
            print(f"[{_now()}] reached --max-hours; stopping.", flush=True)
            break
        # sleep the remainder of the interval, but wake promptly on signal
        wake = t0 + interval_s
        while not _STOP and time.time() < wake:
            time.sleep(min(2.0, wake - time.time()))

    con.close()
    print(f"[{_now()}] stopped after {poll_n} polls. Data in {db}.", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Read-only Moltbook feed snapshotter (2c).")
    ap.add_argument("--db", default="moltbook_snapshots.db", help="SQLite output file")
    ap.add_argument("--interval", type=float, default=20.0,
                    help="minutes between polls (<=30 recommended; ranking half-life is ~3h)")
    ap.add_argument("--sorts", nargs="+", default=SORTS, choices=SORTS)
    ap.add_argument("--pages", type=int, default=3, help="pages per sort (100 posts/page)")
    ap.add_argument("--once", action="store_true", help="single poll then exit (test)")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="auto-stop after this many hours (default: run until Ctrl-C)")
    ap.add_argument("--stats", action="store_true", help="summarize the DB and exit")
    ap.add_argument("--export", metavar="CSV", help="export the DB to CSV and exit")
    args = ap.parse_args()

    if args.stats:
        stats(open_db(args.db)); return
    if args.export:
        export_csv(open_db(args.db), args.export); return
    run(args.db, args.sorts, args.interval, args.pages, args.once, args.max_hours)


if __name__ == "__main__":
    sys.exit(main())
