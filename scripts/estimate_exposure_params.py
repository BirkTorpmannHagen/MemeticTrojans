"""Estimate exposure/visibility parameters from the live feed snapshots (2c).

Fits the pieces the historical single-scrape analysis had to MODEL from the
nominal ranking H (recency half-life 3h + 0.2*upvotes):
  1. Visibility gradient  — upvote-accrual rate vs feed rank (real agents), the
     observational counterpart to the assay's rank->engagement curve.
  2. Engagement decay     — how accrual falls with post age -> effective half-life
     (the real analogue of H's 3h recency half-life). CENSORING-BIASED (see below).
  3. Activity level       — current posts/hr, so the ~4-5x lower-than-corpus traffic
     becomes a measured covariate rather than an unknown confound.

Method: pair consecutive snapshots of the same post under sort='hot' (rank there =
observed visibility). rate = Δupvotes / Δt (upv/hr) at (rank, age). Cross-rank
comparison within snapshots gives the visibility gradient with little censoring
bias; the age->decay term IS censoring-biased (posts leaving top-300 are unobserved,
so surviving old posts are upward-selected) and is reported with that caveat.

    python3 scripts/estimate_exposure_params.py --db moltbook_snapshots.export.db.gz

Writes out/exposure/exposure_params.json.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import sqlite3
import tempfile

import numpy as np
import pandas as pd

OUT = "out/exposure"


def _open(db_path: str) -> sqlite3.Connection:
    if db_path.endswith(".gz"):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        with gzip.open(db_path, "rb") as f:
            shutil.copyfileobj(f, tmp)
        tmp.close()
        return sqlite3.connect(tmp.name)
    return sqlite3.connect(db_path)


def _intervals(con: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT poll_id, snapshot_ts, rank, post_id, upvotes, created_at "
        "FROM snapshots WHERE sort='hot'", con)
    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"], utc=True)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["created_at"]).sort_values(["post_id", "snapshot_ts"]).copy()
    g = df.groupby("post_id", sort=False)
    df["upv2"] = g["upvotes"].shift(-1)
    df["t2"] = g["snapshot_ts"].shift(-1)
    df = df.dropna(subset=["upv2", "t2"])
    df["dt_h"] = (df["t2"] - df["snapshot_ts"]).dt.total_seconds() / 3600.0
    df["dupv"] = (df["upv2"] - df["upvotes"]).clip(lower=0)
    df["age_h"] = (df["snapshot_ts"] - df["created_at"]).dt.total_seconds() / 3600.0
    df = df[(df.dt_h > 0.15) & (df.dt_h < 1.5) & (df.age_h >= 0) & (df["rank"] < 300)]
    df["rate"] = df["dupv"] / df["dt_h"]
    return df


def estimate(db_path: str):
    con = _open(db_path)
    iv = _intervals(con)
    n = len(iv)

    # 1. Visibility gradient: mean accrual rate by rank bin (real-agent rank->engagement)
    bins = [0, 5, 15, 30, 60, 150, 300]
    iv["rank_bin"] = pd.cut(iv["rank"], bins, right=False)
    grad = iv.groupby("rank_bin", observed=True)["rate"].agg(["mean", "count"])
    head = grad["mean"].iloc[0]
    grad["rel_to_head"] = grad["mean"] / head

    # 2. Decay: log(rate) ~ age among HEAD posts (rank<30) to reduce rank confound.
    h = iv[iv["rank"] < 30].copy()
    y = np.log(h["rate"].to_numpy() + 0.1)
    A = np.column_stack([np.ones(len(h)), h["age_h"].to_numpy()])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    b0, b_age = float(coef[0]), float(coef[1])
    half_life_h = float(np.log(2) / -b_age) if b_age < 0 else float("inf")
    baseline_rate = float(np.exp(b0) - 0.1)

    # full OLS (matches handoff form) for reference
    yf = np.log(iv["rate"].to_numpy() + 0.1)
    Af = np.column_stack([np.ones(n), np.log(iv["rank"].to_numpy() + 1), iv["age_h"].to_numpy()])
    cf, *_ = np.linalg.lstsq(Af, yf, rcond=None)

    # 3. Activity now (from 'new' feed created_at span) vs corpus
    nw = pd.read_sql_query("SELECT DISTINCT post_id, created_at FROM snapshots WHERE sort='new'", con)
    nw["created_at"] = pd.to_datetime(nw["created_at"], utc=True, errors="coerce")
    nw = nw.dropna()
    span_h = (nw["created_at"].max() - nw["created_at"].min()).total_seconds() / 3600.0
    posts_per_hr_now = len(nw) / span_h if span_h > 0 else float("nan")
    corpus_avg, corpus_peak = 987.0, 1221.0   # from data_raw (see activity check)
    activity_ratio = corpus_avg / posts_per_hr_now if posts_per_hr_now else float("nan")

    win = pd.read_sql_query("SELECT MIN(snapshot_ts) a, MAX(snapshot_ts) b, "
                            "COUNT(DISTINCT poll_id) p FROM snapshots", con).iloc[0]

    params = {
        "source": db_path,
        "window": [win["a"], win["b"]], "polls": int(win["p"]), "n_intervals": n,
        "visibility_gradient": {str(k): {"rate_upv_hr": round(v["mean"], 3),
                                         "rel_to_head": round(v["rel_to_head"], 3),
                                         "n": int(v["count"])}
                                for k, v in grad.iterrows()},
        "engagement_half_life_h": round(half_life_h, 1),
        "baseline_rate_upv_hr": round(baseline_rate, 3),
        "decay_fit_head": {"b0": round(b0, 4), "b_age": round(b_age, 4), "n": len(h)},
        "ols_logrank": {"b0": round(float(cf[0]), 4), "b_logrank": round(float(cf[1]), 4),
                        "b_age": round(float(cf[2]), 4)},
        "activity": {"posts_per_hr_now": round(posts_per_hr_now, 1),
                     "posts_per_hr_corpus_avg": corpus_avg,
                     "posts_per_hr_corpus_peak": corpus_peak,
                     "activity_ratio_corpus_over_now": round(activity_ratio, 1)},
        "caveats": [
            "half-life is censoring-biased UP: posts leaving top-300 are unobserved, "
            "so surviving old posts are upward-selected (survivorship).",
            "measured at ~%.0fx LOWER activity than the corpus -> real corpus-era "
            "persistence is SHORTER than this (less feed competition now)." % activity_ratio,
            "assumes the ranking algorithm is unchanged since the Feb corpus.",
        ],
    }
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "exposure_params.json")
    json.dump(params, open(path, "w"), indent=2)

    print(f"intervals={n}  polls={params['polls']}  window={win['a']} -> {win['b']}")
    print("\nVISIBILITY GRADIENT (upv/hr by rank):")
    print(grad.round(3).to_string())
    print(f"\nENGAGEMENT half-life (head, rank<30): {half_life_h:.1f} h  "
          f"(baseline {baseline_rate:.2f} upv/hr)  [censoring-biased UP]")
    print(f"OLS log(rate)~log(rank)+age:  logrank {cf[1]:+.3f}  age {cf[2]:+.4f}/hr")
    print(f"\nACTIVITY now: {posts_per_hr_now:.0f} posts/hr  vs corpus ~{corpus_avg:.0f}/hr "
          f"-> {activity_ratio:.1f}x lower now")
    print(f"\nwrote {path}")
    return params


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="moltbook_snapshots.export.db.gz",
                    help="snapshot DB (.db or .gz)")
    args = ap.parse_args()
    estimate(args.db)
