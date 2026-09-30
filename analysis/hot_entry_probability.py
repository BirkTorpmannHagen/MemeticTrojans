"""P(a post reaches the hot / top broadcast feed) from the empirical upvote distribution.

The broadcast surfaces (hot, top) are karma-gated: a post is in the top-K only if its upvotes clear
the K-th-place bar. This estimates the probability a post clears that bar from REAL data:

  1. Empirical **upvote distribution** across all snapshot posts (peak upvotes per post) — the
     "how upvoted do posts get" histogram.
  2. Empirical **entry threshold** tau_s = upvotes of the boundary (rank K-1) post on surface s,
     across polls — the bar to be in that surface's top-K.
  3. **P(reach s) = P(peak_upvotes >= tau_s)** — the tail of the distribution above the bar.
     (Validates against the directly-measured entry rate.)
  4. As a function of a post's **upvote-propensity multiplier** m (a more upvote-worthy post — e.g.
     the memetic Trojan's ~1.3x edge — shifts its whole upvote distribution up by m):
     P(reach s | m) = P(m * U >= tau_s) = P(U >= tau_s / m).
  5. As a function of **posting frequency** F (an attacker seeding F independent posts):
     P(>=1 of F reaches s) = 1 - (1 - p)^F.

Read-only analysis of moltbook_snapshots.export.db.gz (see data/moltbook-feed-snapshots/).

    python -m analysis.hot_entry_probability

Output: figures/hot_entry_probability.pdf, out/attack_reach/hot_entry_probability.csv.
"""
from __future__ import annotations

import gzip
import os
import shutil
import sqlite3
import tempfile

import numpy as np
import pandas as pd

DB_GZ = "moltbook_snapshots.export.db.gz"
# Rolling-window `top` snapshots (top/hour, top/day, top/week), collected separately (read-only)
# via scripts/moltbook_top_windows.py. Gzipped in the repo; falls back to the ungzipped DB if present.
TOPWIN_DB_GZ = "data/moltbook-topwindows-2026-09/moltbook_top_windows.db.gz"
TOPWIN_DB = "moltbook_top_windows.db"
# surface name -> window value in the windowed DB
TOPWIN = {"top_hour": "hour", "top_day": "day", "top_week": "week"}
FIGDIR = "figures"
OUT = "out/attack_reach"
K = 25                       # feed top-K (Moltbook default)
TROJAN_MULT = 1.3            # measured Trojan upvote edge over a generic post (~1.2-1.35x)


def _load():
    tmp = tempfile.mktemp(suffix=".db")
    with gzip.open(DB_GZ, "rb") as f, open(tmp, "wb") as o:
        shutil.copyfileobj(f, o)
    c = sqlite3.connect(tmp)
    df = pd.read_sql_query("select sort,rank,post_id,upvotes,snapshot_ts from snapshots", c)
    c.close(); os.remove(tmp)
    df["upvotes"] = pd.to_numeric(df["upvotes"], errors="coerce").fillna(0)
    return df


def _span_h(ts):
    """Collection span in hours from a snapshot_ts series (>=1e-9)."""
    t = pd.to_datetime(ts, utc=True, errors="coerce").dropna()
    return max((t.max() - t.min()).total_seconds() / 3600.0, 1e-9) if len(t) else 1.0


def _load_windows():
    """Windowed-top snapshots (window in {hour,day,week}); None if not collected yet."""
    if os.path.exists(TOPWIN_DB_GZ):
        tmp = tempfile.mktemp(suffix=".db")
        with gzip.open(TOPWIN_DB_GZ, "rb") as f, open(tmp, "wb") as o:
            shutil.copyfileobj(f, o)
        path, cleanup = tmp, True
    elif os.path.exists(TOPWIN_DB):
        path, cleanup = TOPWIN_DB, False
    else:
        return None
    c = sqlite3.connect(path)
    df = pd.read_sql_query("select window,rank,post_id,upvotes,snapshot_ts from snapshots", c)
    c.close()
    if cleanup:
        os.remove(path)
    df["upvotes"] = pd.to_numeric(df["upvotes"], errors="coerce").fillna(0)
    return df


_CAL = {}


MIN_WINDOW_POLLS = 6         # below this the windowed occupancy is too sparse to estimate timing


def calibration(surface="hot"):
    """(peak_upvotes array, entry bar tau, recency-timing discount) for a surface, cached.
    The discount makes clearing the upvote bar (necessary) match the measured occupancy (sufficient);
    all-time `top` has no recency so discount=1.

    Surfaces:
      * "hot" / "top"  — from the main Aug snapshot DB (all four default sorts).
      * "top_hour"/"top_day"/"top_week" — ROLLING windows: `peak` (the population upvote distribution)
        still comes from the Aug DB, but the entry bar `tau` and occupancy come from the separately
        collected windowed DB (scripts/moltbook_top_windows.py). Raises if that DB is absent."""
    if surface not in _CAL:
        df = _load()
        peak = df.groupby("post_id")["upvotes"].max().to_numpy()
        if surface in TOPWIN:
            wdf = _load_windows()
            if wdf is None:
                raise FileNotFoundError(
                    f"surface {surface!r} needs the windowed-top DB ({TOPWIN_DB_GZ}); collect it with "
                    "scripts/moltbook_top_windows.py --param time --windows hour day week")
            w = TOPWIN[surface]
            sub = wdf[wdf["window"] == w]
            b = sub[sub["rank"] == K - 1]["upvotes"]
            tau = float(b.median()) if len(b) else float("nan")
            p_tail = float((peak >= tau).mean())
            npolls = sub["snapshot_ts"].nunique()
            if npolls >= MIN_WINDOW_POLLS and p_tail > 0:
                # occupancy fraction, duration-normalized: the windowed collection and the Aug
                # population span different hours, so scale the population count to the windowed span
                # (post rate is ~constant, collections are ~3 weeks apart) before taking the ratio.
                occ = sub[sub["rank"] < K]["post_id"].nunique()
                pop_rate = len(peak) / _span_h(df["snapshot_ts"])         # Aug posts per hour
                exp_pop = max(pop_rate * _span_h(sub["snapshot_ts"]), 1.0)  # expected posts over window span
                p_meas = occ / exp_pop
                timing = min(1.0, p_meas / p_tail)
            else:
                timing = 1.0          # too few polls to estimate occupancy: rolling bar ~ sufficient
        else:
            b = df[(df["sort"] == surface) & (df["rank"] == K - 1)]["upvotes"]
            tau = float(b.median()) if len(b) else float("nan")
            p_tail = float((peak >= tau).mean())
            p_meas = df[(df["sort"] == surface) & (df["rank"] < K)]["post_id"].nunique() / len(peak)
            timing = min(1.0, p_meas / p_tail) if p_tail > 0 else 1.0
        _CAL[surface] = (peak, tau, timing)
    return _CAL[surface]


def p_reach(m=1.0, F=1, surface="hot"):
    """Calibrated P(a post/attacker reaches `surface`'s top-K broadcast): a post with upvote-propensity
    multiplier m clears the bar with prob p1 = P(m*U >= tau) * recency-timing; F independent posts give
    1-(1-p1)^F. m=1 reproduces the measured base entry rate."""
    peak, tau, timing = calibration(surface)
    p1 = min(1.0, float((peak * m >= tau).mean()) * timing)
    return 1.0 - (1.0 - p1) ** F


def run():
    os.makedirs(FIGDIR, exist_ok=True); os.makedirs(OUT, exist_ok=True)
    df = _load()
    # peak upvotes per DISTINCT post (its best shot at the bar)
    peak = df.groupby("post_id")["upvotes"].max().to_numpy()
    npost = len(peak)
    print(f"{npost} distinct posts; peak upvotes: median={np.median(peak):.0f} "
          f"p90={np.percentile(peak,90):.0f} p99={np.percentile(peak,99):.0f} max={peak.max():.0f}")

    # surfaces: the two default sorts, plus the rolling-top windows if their DB has been collected
    surfaces = ["hot", "top"]
    if _load_windows() is not None:
        surfaces += list(TOPWIN)                                   # top_hour, top_day, top_week
    else:
        print("\n(rolling-top windows not collected yet; run scripts/moltbook_top_windows.py to add "
              "top_hour/top_day/top_week)")

    rows = []
    for s in surfaces:
        _, tau, timing = calibration(s)                            # peak is the shared population
        p_tail = float((peak >= tau).mean())                       # P(clears the upvote bar)
        print(f"\n{s}: entry bar tau (rank {K-1} upvotes, median over polls) = {tau:.0f}")
        print(f"  P(clears upvote bar) = {p_tail*100:.2f}%  ->  x recency-timing discount {timing:.2f}  "
              f"= P(occupy {s}) {p_tail*timing*100:.2f}%")
        for m in [1.0, TROJAN_MULT, 1.6, 2.0, 3.0]:
            rows.append(dict(surface=s, tau=tau, mult=m, freq=1, timing=round(timing, 3),
                             p_reach=round(p_reach(m, 1, s), 4)))
        for F in [1, 3, 10, 30, 100]:
            rows.append(dict(surface=s, tau=tau, mult=TROJAN_MULT, freq=F, timing=round(timing, 3),
                             p_reach=round(p_reach(TROJAN_MULT, F, s), 4)))
        print(f"  Trojan (x{TROJAN_MULT}) single post: {p_reach(TROJAN_MULT,1,s)*100:.2f}%  | seeding F posts: "
              + ", ".join(f"F={F}:{p_reach(TROJAN_MULT,F,s)*100:.1f}%" for F in [3, 10, 30, 100]))

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "hot_entry_probability.csv"), index=False)
    _plot(peak, df)
    print(f"\nwrote {OUT}/hot_entry_probability.csv, {FIGDIR}/hot_entry_probability.pdf")


def _plot(peak, df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    # (a) upvote distribution + entry bars
    ax1.hist(np.clip(peak, 0, 300), bins=60, color="#4c78a8", alpha=0.85)
    ax1.set_yscale("log")
    for s, col in [("hot", "#e45756"), ("top", "#f58518")]:
        b = df[(df["sort"] == s) & (df["rank"] == K - 1)]["upvotes"]
        if len(b):
            tau = float(b.median()); ax1.axvline(tau, color=col, lw=2, ls="--")
            ax1.text(tau, ax1.get_ylim()[1] * 0.5, f" {s} bar ~{tau:.0f}", color=col, fontsize=8, rotation=90, va="top")
    ax1.set_xlabel("peak upvotes per post"); ax1.set_ylabel("# posts (log)")
    ax1.set_title("Empirical upvote distribution vs broadcast-entry bar", fontsize=10)
    ax1.spines[["top", "right"]].set_visible(False)
    # (b) P(reach hot) vs propensity multiplier and posting frequency (recency-timing calibrated)
    tau_hot = float(df[(df["sort"] == "hot") & (df["rank"] == K - 1)]["upvotes"].median())
    p_tail_hot = float((peak >= tau_hot).mean())
    p_meas_hot = df[(df["sort"] == "hot") & (df["rank"] < K)]["post_id"].nunique() / len(peak)
    timing = min(1.0, p_meas_hot / p_tail_hot) if p_tail_hot > 0 else 1.0
    mults = np.linspace(1.0, 3.0, 40)
    ax2.plot(mults, [min(1.0, (peak * m >= tau_hot).mean() * timing) for m in mults], color="#c0392b", lw=2,
             label="single post, vs upvote-propensity ×m")
    pT = min(1.0, float((peak * TROJAN_MULT >= tau_hot).mean()) * timing)
    Fs = np.arange(1, 101)
    ax2b = ax2.twiny()
    ax2b.plot(Fs, 1 - (1 - pT) ** Fs, color="#2c3e50", lw=2, ls="--",
              label=f"Trojan (×{TROJAN_MULT}), vs # posts F")
    ax2.axvline(TROJAN_MULT, color="#178", ls=":", lw=1.2); ax2.text(TROJAN_MULT, 0.5, " Trojan ×1.3", color="#178", fontsize=8, rotation=90)
    ax2.set_xlabel("upvote-propensity multiplier m"); ax2.set_ylabel("P(reach hot broadcast feed)")
    ax2b.set_xlabel("attacker posting frequency F (Trojan propensity)", color="#2c3e50")
    ax2.set_title("P(reach hot) vs post quality and posting frequency", fontsize=10)
    ax2.spines[["top"]].set_visible(False)
    ax2.legend(fontsize=8, loc="upper left"); ax2b.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "hot_entry_probability.pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
