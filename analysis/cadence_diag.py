"""Diagnose the sub-heartbeat coincidence spike that pins the naive Hawkes MLE at
the grid floor. Tests the operator-hypothesis (Li 2026, arXiv:2602.07432): are the
near-simultaneous, DIFFERENT-agent post pairs produced by off-heartbeat, operator-
driven / bot-farmed accounts rather than genuine agent-to-agent re-transmission?

Reports:
  1. Timestamp resolution (are created_at quantized -> artificial coincidences?).
  2. Per-author cadence fingerprint: CoV of inter-post intervals. Li's cut:
     CoV<0.5 autonomous (regular heartbeat), CoV>1.0 human-influenced (irregular).
  3. Concentration: do a few accounts dominate posting (bot-farm signature)?
  4. Spike composition: of the <5min / <30min DIFFERENT-author pairs in the pooled
     seeded meme-event series, what share touch a high-CoV (operator) author?

    python -m analysis.cadence_diag
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from analysis.endogenous_hawkes import (
    MIN_EVENTS, load_marked_corpus, get_units, _pattern, meme_events)

MIN_POSTS_COV = 5     # min posts to fingerprint an author's cadence
HEARTBEAT_H = 0.5


def author_cadence(corpus):
    """Per-author CoV of inter-post intervals (hours) + post count."""
    rows = []
    for a, g in corpus.groupby("author_name"):
        t = np.sort((g["created_at"].astype("int64").to_numpy()) / 3.6e12)  # ns->h
        if len(t) < MIN_POSTS_COV:
            continue
        d = np.diff(t)
        d = d[d > 0]
        if len(d) < 2 or d.mean() == 0:
            continue
        rows.append((a, len(t), float(d.std() / d.mean()), float(np.median(d))))
    return pd.DataFrame(rows, columns=["author", "n_posts", "cov", "median_gap_h"])


def run():
    print("[1/4] loading corpus ...", flush=True)
    corpus, post_text_by_id = load_marked_corpus()
    corpus = corpus.sort_values("created_at").reset_index(drop=True)
    t0 = corpus["created_at"].min()

    # 1. timestamp resolution
    ts = corpus["created_at"].astype("int64").to_numpy()
    allgaps = np.diff(np.sort(ts)) / 1e9  # seconds
    nz = allgaps[allgaps > 0]
    sec = pd.to_datetime(corpus["created_at"]).dt.second.to_numpy()
    print(f"\n[2/4] TIMESTAMP RESOLUTION")
    print(f"      exact-duplicate timestamps: {100*np.mean(allgaps==0):.1f}% of consecutive gaps == 0")
    print(f"      min non-zero gap: {nz.min():.3f}s  | frac seconds-field==0: {np.mean(sec==0):.2f}")
    print(f"      gaps <1s: {100*np.mean(allgaps<1):.1f}%  <60s: {100*np.mean(allgaps<60):.1f}%")

    # 2 + 3. cadence fingerprint + concentration
    cad = author_cadence(corpus)
    auto = cad[cad["cov"] < 0.5]; human = cad[cad["cov"] > 1.0]
    cov_by_author = dict(zip(cad["author"], cad["cov"]))
    postcount = corpus["author_name"].value_counts()
    top4 = postcount.head(4).sum()
    print(f"\n[3/4] CADENCE FINGERPRINT (Li 2026 CoV cut)  [{len(cad)} authors >= {MIN_POSTS_COV} posts]")
    print(f"      autonomous (CoV<0.5): {len(auto):4d} authors ({100*len(auto)/len(cad):.1f}%)")
    print(f"      human-infl (CoV>1.0): {len(human):4d} authors ({100*len(human)/len(cad):.1f}%)")
    print(f"      median CoV: {cad['cov'].median():.2f}  | median gap: {cad['median_gap_h'].median():.2f}h "
          f"(heartbeat ~{HEARTBEAT_H}h)")
    print(f"      CONCENTRATION: top-4 authors = {100*top4/len(corpus):.1f}% of all utterances")

    # 4. spike composition on the pooled seeded event series
    print(f"\n[4/4] SPIKE COMPOSITION (per-author-first events, all pairs DIFFERENT-author)")
    # rebuild per-author-first events WITH author identity, pooled over seeded units
    near5=near30=tot30=0; near5_op=near30_op=0
    for label, family, _tier in get_units(smoke=False):
        name = label.split("] ", 1)[-1] if "] " in label else label
        kind = family if family in ("phrase","hashtag","domain","coined") else "emoji"
        pat = _pattern(name, kind)
        mask = corpus["text"].str.contains(pat, regex=True, na=False)
        hits = corpus.loc[mask, ["author_name","created_at"]]
        if len(hits) < MIN_EVENTS:
            continue
        first = hits.sort_values("created_at").groupby("author_name", sort=False).first().reset_index()
        first = first.sort_values("created_at")
        th = (first["created_at"] - t0).dt.total_seconds().to_numpy()/3600.0
        au = first["author_name"].to_numpy()
        covs = np.array([cov_by_author.get(a, np.nan) for a in au])
        lo = np.searchsorted(th, th - 0.5)  # 30-min window
        for i in range(1, len(th)):
            for j in range(lo[i], i):
                d = th[i]-th[j]
                if d > 0.5: continue
                tot30 += 1
                op = (covs[i] > 1.0) or (covs[j] > 1.0)   # touches an operator-class author
                if d <= 5/60: near5 += 1; near5_op += op
                if d <= 0.5:  near30 += 1; near30_op += op
    def pct(a,b): return f"{100*a/max(b,1):.0f}%"
    print(f"      different-author pairs <=5min : {near5:6d}  ({pct(near5_op,near5)} touch an operator-class author)")
    print(f"      different-author pairs <=30min: {near30:6d}  ({pct(near30_op,near30)} touch an operator-class author)")
    print(f"\nSUMMARY: if the <5min pairs are mostly operator-class, the MLE floor is an "
          f"operator/coordination artifact, and a cadence filter should clean tau.")


if __name__ == "__main__":
    run()
