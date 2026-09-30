"""EXPLORATORY (probably unused): upvote ATTRACTION per unit, on the same units as the contagions
table. Complement to the transmission metrics (R_endo, phi_e): instead of "does carrying this unit
predict onward posting?", we ask "do real posts carrying this unit attract more upvotes?".

Observational, data-only: for each unit (coined / lexical / phrase / bge-large+HDBSCAN cluster), take
the REAL Moltbook posts carrying it and summarise their upvote distribution against the corpus
baseline. Joined with R_endo / phi_e from out/endogenous_contagion_strict.csv so attraction can be
compared to transmission (e.g. consciousness: high attraction, fails the phi_e transmission placebo).

    PYTHONPATH=. python -m analysis.upvote_attraction.attraction

Outputs (all in this directory): upvote_attraction.csv, fig_attraction_vs_transmission.pdf.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from analysis.endogenous_hawkes import load_marked_corpus, get_units, _pattern

HERE = os.path.dirname(os.path.abspath(__file__))
CONTAGION = "out/endogenous_contagion_strict.csv"
CLUSTER_NPY = "out/cluster_v2/assignment_mcs200.npy"
CLUSTER_LAB = "out/cluster_v2/labels_mcs200.csv"


def _stats(upv, base_mean, hot_bar, rest=None):
    upv = np.asarray(upv, dtype="float64")
    n = len(upv)
    if n == 0:
        return None
    # one-sided test that carrier posts are upvoted above the rest of the corpus (r_up > 1):
    # Mann-Whitney U is robust to the heavy-tailed upvote distribution. p_rup is the significance
    # of the observed upvote edge.
    p_rup = float("nan")
    if rest is not None and n >= 2 and len(rest) >= 2:
        from scipy.stats import mannwhitneyu
        try:
            p_rup = float(mannwhitneyu(upv, np.asarray(rest, "float64"),
                                       alternative="greater", method="asymptotic").pvalue)
        except ValueError:                       # all-equal inputs
            p_rup = 1.0
    return dict(n_posts=n, mean_upv=round(float(upv.mean()), 3),
                median_upv=float(np.median(upv)), p90_upv=float(np.percentile(upv, 90)),
                upv_lift=round(float(upv.mean() / base_mean), 3) if base_mean else float("nan"),
                frac_ge_hotbar=round(float((upv >= hot_bar).mean()), 4),
                p_rup=round(p_rup, 5) if p_rup == p_rup else float("nan"))


def run():
    corpus, _ = load_marked_corpus(posts_only=True, cluster_npy=CLUSTER_NPY)
    upv_all = corpus["upvotes"].to_numpy(dtype="float64")
    base_mean = float(upv_all.mean())
    hot_bar = float(np.percentile(upv_all, 99))   # ~top-1% upvote count: a proxy "attention" threshold
    print(f"corpus: {len(corpus)} posts | mean upvotes {base_mean:.2f} | median {np.median(upv_all):.0f} "
          f"| p99 (hot-bar proxy) {hot_bar:.0f}", flush=True)

    rows = []
    for label, family, tier in get_units():
        disp = label.split("] ", 1)[-1] if label.startswith("[") else label
        pat = _pattern(disp, family if family in ("phrase", "hashtag", "domain", "coined") else "emoji")
        mask = corpus["text"].str.contains(pat, regex=True, na=False).to_numpy()
        st = _stats(corpus["upvotes"].to_numpy()[mask], base_mean, hot_bar,
                    rest=corpus["upvotes"].to_numpy()[~mask])
        if st and st["n_posts"] >= 40:
            rows.append(dict(unit=label, tier=tier, **st))

    # semantic clusters (same bge-large + HDBSCAN assignment as the contagions table)
    if os.path.exists(CLUSTER_LAB):
        labs = pd.read_csv(CLUSTER_LAB); cl = corpus["cluster"].to_numpy()
        uv = corpus["upvotes"].to_numpy()
        for r in labs.itertuples():
            m = cl == int(r.cluster)
            st = _stats(uv[m], base_mean, hot_bar, rest=uv[~m])
            if st and st["n_posts"] >= 40:
                rows.append(dict(unit=f"[cluster] {r.label}", tier="cluster", **st))

    df = pd.DataFrame(rows)
    # join transmission metrics for the attraction-vs-transmission comparison
    ct = pd.read_csv(CONTAGION)[["unit", "R_endo", "phi_endo"]]
    df = df.merge(ct, on="unit", how="left").sort_values("upv_lift", ascending=False).reset_index(drop=True)
    out_csv = os.path.join(HERE, "upvote_attraction.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv} ({len(df)} units)\n")
    print("TOP upvote-attracting units:")
    print(df.head(20)[["unit", "tier", "n_posts", "mean_upv", "upv_lift", "frac_ge_hotbar",
                       "R_endo", "phi_endo"]].to_string(index=False))
    _plot(df, os.path.join(HERE, "fig_attraction_vs_transmission.pdf"))


def _plot(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = df.dropna(subset=["phi_endo"])
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"coined": "#8c2d04", "lexical": "#5a5a5a", "phrase": "#2a6f97", "cluster": "#2a8f6b"}
    for t, g in d.groupby("tier"):
        ax.scatter(g["phi_endo"], g["upv_lift"], s=18, alpha=0.7,
                   c=colors.get(t, "#888"), label=t)
    ax.axvline(0, color="crimson", lw=1, ls="--")
    ax.axhline(1, color="0.6", lw=1, ls=":")
    ax.set_xlabel(r"$\phi_e$  (transmission placebo; $>0$ = forward-directional)")
    ax.set_ylabel("upvote lift  (unit mean / corpus mean)")
    ax.set_title("Attraction (upvotes) vs transmission ($\\phi_e$) per unit")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    run()
