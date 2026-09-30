# Collect: rolling-window `top` snapshots (top/hour, top/day, top/week)

**Run this on the always-on desktop** (the laptop is intermittently off). Read-only: it only does
HTTP GET to the public `https://moltbook.com/api/v1/posts` endpoint — it never posts, votes,
comments, or authenticates. Stdlib only (Python 3.8+), no venv needed.

## Why
The original snapshots (`data/moltbook-feed-snapshots/`) polled `sort=top` with no time window, so we
only captured the **all-time** top — a frozen leaderboard nothing new enters. Moltbook's `time=` query
param exposes genuine rolling windows (`hour ⊂ day ⊂ week ⊂ month ⊂ year ⊂ all`, confirmed by
`--probe`), which are *refreshing* broadcast surfaces with far lower entry bars (top/hour ~9 upvotes,
top/day ~71, vs the `hot` bar ~111 and all-time top ~1,334+). We model top/hour, top/day, top/week as
additional state-mediated surfaces; this collects their real entry bars and turnover.

## Steps

1. On the desktop, fetch and check out the collection branch:
   ```bash
   git fetch origin
   git checkout topwindows-collect     # branch carrying the collector + analysis
   ```

2. (Optional) re-confirm the window param still works (~15 s, read-only):
   ```bash
   python3 scripts/moltbook_top_windows.py --probe
   ```
   Expect: `Genuine rolling top-window param: 'time'`.

3. Start the collection. **~24 h at 20-min polls is the sweet spot** (robust entry bars for all three
   windows + full hour/day turnover). Writes to `moltbook_top_windows.db` in the repo root:
   ```bash
   python3 scripts/moltbook_top_windows.py --param time --windows hour day week --interval 20 --max-hours 24
   ```
   - Leave it in a terminal, or background it:
     `nohup python3 scripts/moltbook_top_windows.py --param time --windows hour day week --interval 20 --max-hours 24 > topwin.log 2>&1 &`
   - **Intermittent off is fine.** It appends to one SQLite file (WAL, restart-safe). If the machine
     sleeps or you stop it (Ctrl-C), just re-run the same command — it keeps appending to the same DB.
     `--max-hours` counts wall-clock *while running*, so across restarts just re-run until you have
     enough polls.
   - Want crisper top/hour dwell (it turns over in ≤1 h)? add `--interval 10`.

4. Check what you've got any time:
   ```bash
   python3 scripts/moltbook_top_windows.py --stats
   ```
   Aim for **≥ ~40–70 polls** (the analysis needs ≥6 per window to estimate the timing discount; more
   is better). Each window should show a few hundred distinct posts.

5. When done, drop the DB into this folder, gzip it, and push:
   ```bash
   gzip -c moltbook_top_windows.db > data/moltbook-topwindows-2026-09/moltbook_top_windows.db.gz
   git add data/moltbook-topwindows-2026-09/moltbook_top_windows.db.gz
   git commit -m "Add rolling-window top snapshots (top/hour,day,week)"
   git push
   ```
   The analysis (`analysis/hot_entry_probability.py`) already looks for the gz at exactly
   `data/moltbook-topwindows-2026-09/moltbook_top_windows.db.gz` and auto-includes the three windows
   once it's present. Nothing else to wire on your end — ping me on the laptop after you push and I'll
   regenerate the tables.

## Notes
- The collector's DB schema mirrors the original snapshotter plus a `window` column
  (`hour`/`day`/`week`), so the existing entry/persistence analysis reads it directly.
- In a quiet week, top/day and top/week can coincide (same top-25); that's the real platform state,
  not a bug — top/day is then the operative easier bar.
