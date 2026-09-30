#!/usr/bin/env python3
"""Read-only collector for TIME-WINDOWED `top` feeds (top/day, top/week).

WHY: the original snapshotter (scripts/moltbook_snapshotter.py) polled `sort=top` with NO time
window, so it captured the *all-time* top — a frozen high-karma leaderboard nothing new enters
(see data/moltbook-feed-snapshots/COLLECTION.md). Reddit-style platforms also expose rolling
windows (top today / this week), which are *refreshing* surfaces with a far lower entry bar and
thus a plausible payload-attack surface. This script probes whether Moltbook honors a window
parameter and, if so, snapshots top/day and top/week over time so they can be modeled like `hot`.

SAFETY / SCOPE: strictly READ-ONLY. Only HTTP GET to the public https://moltbook.com/api/v1/posts
endpoint. NEVER posts, votes, comments, authenticates, or calls any write endpoint. Stdlib only
(runs on the desktop poller with no venv), same UA/politeness as the snapshotter.

STEP 1 — confirm the API honors a window param (which spelling):
    python3 scripts/moltbook_top_windows.py --probe

STEP 2 — collect (probe confirmed the param is `time`; windows default to hour/day/week):
    python3 scripts/moltbook_top_windows.py --param time --windows hour day week --interval 20 --max-hours 24
    python3 scripts/moltbook_top_windows.py --param time --once         # single test poll
    python3 scripts/moltbook_top_windows.py --stats                     # summarize DB

Data -> ./moltbook_top_windows.db (override --db), schema mirrors the snapshotter plus a `window`
column, so analysis/feed_surface_mechanics-style entry/persistence math drops straight in.
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
DEFAULT_WINDOWS = ["hour", "day", "week"]  # rolling top windows to snapshot (month/year deprioritized)
PROBE_KEYS = ["t", "timeframe", "period", "range", "time", "window"]
PAGE_LIMIT = 100
REQUEST_DELAY_S = 0.7
TIMEOUT_S = 25
CONTENT_CHARS = 6000

_STOP = False


def _on_signal(signum, frame):
    global _STOP
    _STOP = True
    print(f"\n[{_now()}] signal {signum} — finishing current poll, then exiting.", flush=True)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# Fetch (read-only GET)                                                        #
# --------------------------------------------------------------------------- #
def _get(params: dict, retries: int = 4):
    url = f"{API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                return url, json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001 — transient errors must not kill the loop
            last = e
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} tries: {url} :: {last}")


def _posts(data):
    return (data.get("posts") if isinstance(data, dict) else data) or []


# --------------------------------------------------------------------------- #
# Probe: does the API honor a window param, and under which spelling?          #
# --------------------------------------------------------------------------- #
def _summ(data):
    now = datetime.now(timezone.utc)
    ids, ups, ages = [], [], []
    for p in _posts(data):
        ids.append(p.get("id"))
        ups.append(p.get("upvotes") or 0)
        ca = p.get("created_at")
        if ca:
            try:
                ages.append((now - datetime.fromisoformat(ca.replace("Z", "+00:00"))).days)
            except Exception:
                pass
    return ids, ups, ages


def _jac(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(len(sa | sb), 1)


def _fetch_summ(params):
    _, d = _get(params)
    return _summ(d)


def _line(label, ids, ups, ages, refs):
    j = "  ".join(f"J({name})={_jac(ids, rid):.2f}" for name, rid in refs)
    print(f"  {label:22s} n={len(ids):3d}  up {min(ups or [0]):>5}-{max(ups or [0]):<5}  "
          f"age {min(ages or [0]):>3}-{max(ages or [0]):<3}d  {j}")
    return ids, ups, ages


def probe():
    """Discriminating probe: a *real* rolling window must (a) differ from all-time top, (b) NOT be a
    mere alias of `new`/`hot`, and (c) vary with the window value (week strictly wider than day:
    higher max-upvotes, older max-age). An inert value that returns one fixed feed regardless fails (c)."""
    print("READ-ONLY probe: does sort=top honor a genuine time WINDOW (vs ignoring the value)?\n")
    # anchors
    top_ids, top_up, top_age = _fetch_summ({"sort": "top", "limit": 25}); time.sleep(REQUEST_DELAY_S)
    new_ids, _, _ = _fetch_summ({"sort": "new", "limit": 25}); time.sleep(REQUEST_DELAY_S)
    hot_ids, _, _ = _fetch_summ({"sort": "hot", "limit": 25}); time.sleep(REQUEST_DELAY_S)
    refs = [("top", top_ids), ("new", new_ids), ("hot", hot_ids)]
    print(f"  {'ANCHOR sort=top':22s} n={len(top_ids):3d}  up {min(top_up or [0])}-{max(top_up or [0])}"
          f"  age {min(top_age or [0])}-{max(top_age or [0])}d   (all-time leaderboard)")
    print(f"  {'ANCHOR sort=new':22s} n={len(new_ids)}   |   {'ANCHOR sort=hot':22s} n={len(hot_ids)}\n")

    # sweep the canonical window ladder under each candidate param
    ladder = ["hour", "day", "week", "month", "year", "all"]
    got = {}   # (key,val) -> (ids, ups, ages)
    for key in ["time", "t", "timeframe", "period", "range", "sort_time", "top"]:
        print(f"--- param '{key}' ---")
        for val in ladder:
            try:
                ids, ups, ages = _fetch_summ({"sort": "top", key: val, "limit": 25})
            except Exception as e:
                print(f"  {key}={val:9s} ERROR: {e}"); time.sleep(REQUEST_DELAY_S); continue
            _line(f"{key}={val}", ids, ups, ages, refs)
            got[(key, val)] = (ids, ups, ages)
            time.sleep(REQUEST_DELAY_S)

    # A genuine rolling window forms a monotone ladder: as it widens, max-upvotes are
    # non-decreasing and the widest (year/all) lands on the all-time leaderboard. Adjacent windows
    # may coincide in a quiet period (e.g. day==week), so we test the whole ladder, not one pair.
    print("\n=== VERDICT ===")
    real = []
    for key in ["time", "t", "timeframe", "period", "range", "sort_time", "top"]:
        seq = [got.get((key, v)) for v in ladder]
        if any(s is None for s in seq):
            continue
        maxup = [max(s[1] or [0]) for s in seq]
        distinct = len({frozenset(s[0]) for s in seq})
        monotone = all(maxup[i] <= maxup[i + 1] + 1 for i in range(len(maxup) - 1))
        widest_is_alltime = _jac(seq[-1][0], top_ids) > 0.9
        fresh_start = max(seq[0][2] or [99]) <= 2 and maxup[0] < maxup[-1]
        if distinct >= 3 and monotone and widest_is_alltime and fresh_start:
            print(f"  '{key}': REAL WINDOW — {distinct} distinct feeds, max-upvotes {maxup} "
                  f"(monotone), widest==all-time top, narrowest is fresh.")
            real.append(key)
        else:
            print(f"  '{key}': not a window (distinct={distinct}, monotone={monotone}, "
                  f"widest==top:{widest_is_alltime}, fresh_start:{fresh_start}).")
    print()
    if real:
        k = real[0]
        # entry bar per window = min upvotes among the top-25 (the 25th slot)
        bars = {v: (min(got[(k, v)][1]) if got[(k, v)][1] else None) for v in ladder}
        print(f"==> Genuine rolling top-window param: '{k}'. Per-window top-25 entry bar (min upvotes): "
              + ", ".join(f"{v}={bars[v]}" for v in ladder if bars[v] is not None))
        print(f"    Collect day+week over time (for dwell + entry-bar series):")
        print(f"    python3 scripts/moltbook_top_windows.py --param {k} --windows day week "
              f"--interval 20 --max-hours 48")
    else:
        print("==> No genuine rolling top-window. `top` is all-time only; model `hot` as the sole "
              "dynamic state surface.")


# --------------------------------------------------------------------------- #
# Storage                                                                      #
# --------------------------------------------------------------------------- #
DDL = """
CREATE TABLE IF NOT EXISTS snapshots (
    poll_id       INTEGER NOT NULL,
    snapshot_ts   TEXT    NOT NULL,
    sort          TEXT    NOT NULL,   -- always 'top' here
    window        TEXT    NOT NULL,   -- day | week
    rank          INTEGER NOT NULL,
    post_id       TEXT    NOT NULL,
    author_id     TEXT, author_name TEXT, author_karma INTEGER, author_followers INTEGER,
    submolt_name  TEXT, upvotes INTEGER, downvotes INTEGER, score INTEGER, hot_score REAL,
    comment_count INTEGER, verification_status TEXT, is_pinned INTEGER,
    created_at    TEXT, title TEXT, content TEXT,
    PRIMARY KEY (poll_id, window, post_id)
);
CREATE INDEX IF NOT EXISTS idx_tw_post_time ON snapshots (post_id, snapshot_ts);
CREATE INDEX IF NOT EXISTS idx_tw_win ON snapshots (window, poll_id);
"""

INSERT = """INSERT OR IGNORE INTO snapshots
(poll_id, snapshot_ts, sort, window, rank, post_id, author_id, author_name, author_karma,
 author_followers, submolt_name, upvotes, downvotes, score, hot_score, comment_count,
 verification_status, is_pinned, created_at, title, content)
VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""


def open_db(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path, timeout=30)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.executescript(DDL)
    con.commit()
    return con


def _row(post, window, rank, poll_id, ts):
    a = post.get("author") or {}
    sm = post.get("submolt") or {}
    return (poll_id, ts, "top", window, rank, post.get("id"),
            post.get("author_id"), a.get("name"), a.get("karma"), a.get("followerCount"),
            sm.get("name"), post.get("upvotes"), post.get("downvotes"), post.get("score"),
            post.get("hot_score"), post.get("comment_count"), post.get("verification_status"),
            1 if post.get("is_pinned") else 0, post.get("created_at"),
            post.get("title"), (post.get("content") or "")[:CONTENT_CHARS])


def collect_window(param, window, poll_id, ts, max_pages):
    rows, cursor, rank = [], None, 0
    for _ in range(max_pages):
        params = {"sort": "top", param: window, "limit": PAGE_LIMIT}
        if cursor:
            params["cursor"] = cursor
        _, data = _get(params)
        posts = _posts(data)
        if not posts:
            break
        for p in posts:
            rows.append(_row(p, window, rank, poll_id, ts)); rank += 1
        if not data.get("has_more"):
            break
        cursor = data.get("next_cursor")
        if not cursor:
            break
        time.sleep(REQUEST_DELAY_S)
    return rows


def poll_once(con, param, windows, max_pages):
    poll_id, ts, total = int(time.time()), _now(), 0
    for win in windows:
        try:
            rows = collect_window(param, win, poll_id, ts, max_pages)
        except Exception as e:  # noqa: BLE001
            print(f"[{ts}]   top/{win} FAILED: {e}", flush=True); continue
        con.executemany(INSERT, rows)
        total += len(rows)
        print(f"[{ts}]   top/{win:5s} {len(rows):4d} posts", flush=True)
        time.sleep(REQUEST_DELAY_S)
    con.commit()
    return total


def stats(con):
    q = con.execute("SELECT COUNT(*), COUNT(DISTINCT poll_id), COUNT(DISTINCT post_id), "
                    "MIN(snapshot_ts), MAX(snapshot_ts) FROM snapshots").fetchone()
    rows, polls, posts, tmin, tmax = q
    if not rows:
        print("no snapshots yet."); return
    print(f"{rows} rows | {polls} polls | {posts} distinct posts | {tmin} -> {tmax}")
    for (win,) in con.execute("SELECT DISTINCT window FROM snapshots ORDER BY window"):
        n = con.execute("SELECT COUNT(DISTINCT post_id) FROM snapshots WHERE window=?",
                        (win,)).fetchone()[0]
        print(f"  top/{win}: {n} distinct posts")


def run(db, param, windows, interval_min, max_pages, once, max_hours):
    con = open_db(db)
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    deadline = time.time() + max_hours * 3600 if max_hours else None
    print(f"[{_now()}] top-window collector | db={db} | param={param} | windows={windows} | "
          f"every {interval_min} min" + (f" | stop after {max_hours}h" if max_hours else ""),
          flush=True)
    poll_n = 0
    while not _STOP:
        t0 = time.time()
        try:
            n = poll_once(con, param, windows, max_pages); poll_n += 1
            print(f"[{_now()}] poll #{poll_n}: {n} rows committed.", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{_now()}] poll error (continuing): {e}", flush=True)
        if once or _STOP:
            break
        if deadline and time.time() >= deadline:
            print(f"[{_now()}] reached --max-hours; stopping.", flush=True); break
        wake = t0 + interval_min * 60.0
        while not _STOP and time.time() < wake:
            time.sleep(min(2.0, wake - time.time()))
    con.close()
    print(f"[{_now()}] stopped after {poll_n} polls. Data in {db}.", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Read-only Moltbook top/day+top/week collector.")
    ap.add_argument("--probe", action="store_true", help="test which window param the API honors, then exit")
    ap.add_argument("--param", default="time", help="window param confirmed by --probe (default time)")
    ap.add_argument("--windows", nargs="+", default=DEFAULT_WINDOWS,
                    help=f"rolling windows to snapshot (default {' '.join(DEFAULT_WINDOWS)})")
    ap.add_argument("--db", default="moltbook_top_windows.db")
    ap.add_argument("--interval", type=float, default=20.0, help="minutes between polls")
    ap.add_argument("--pages", type=int, default=3, help="pages per window (100 posts/page)")
    ap.add_argument("--once", action="store_true", help="single poll then exit (test)")
    ap.add_argument("--max-hours", type=float, default=None, help="auto-stop after this many hours")
    ap.add_argument("--stats", action="store_true", help="summarize the DB and exit")
    args = ap.parse_args()

    if args.probe:
        probe(); return
    if args.stats:
        stats(open_db(args.db)); return
    run(args.db, args.param, args.windows, args.interval, args.pages, args.once, args.max_hours)


if __name__ == "__main__":
    sys.exit(main())
