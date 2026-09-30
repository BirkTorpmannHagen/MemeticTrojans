"""Data-driven estimation of the Hawkes excitation timescale tau (the re-emission
lag), instead of fixing it. Two independent estimators, per the Hawkes literature:

  (A) NON-PARAMETRIC empirical kernel (Marsan-Lengline-flavoured declustering).
      For every unit, pool all within-24h pairwise lags d = t_i - t_j (j<i).
      NULL = within-day uniform reshuffle: each event redrawn uniformly inside its
      own 24h bin, so per-day counts (the exogenous daily background / growth trend)
      are PRESERVED but sub-day clustering is destroyed. excess(d) = obs(d) - null(d)
      is the triggering kernel at sub-day resolution, controlling for the daily
      background. Its half-life is a background-free, assumption-light tau_hat.
      This directly tests the "one-heartbeat floor, few-heartbeat decay" hypothesis.

  (B) PROFILE LIKELIHOOD (Ozaki/Ogata standard, made robust). For each unit fit
      (mu_d, alpha) by MLE at each tau on a grid; the fit's LL is the profile
      log-likelihood L_unit(tau). Pool across units -> L(tau); tau_hat = argmax,
      with a 2-LL-unit interval. Free per-day background (the primary spec).

If (A) peaks at the heartbeat scale while (B) drifts larger, that is the classic
branching-ratio/non-stationarity aliasing (Filimonov-Sornette): a broad kernel
absorbs slow trend. We then bound tau at the empirically-resolved scale, floored at
the heartbeat (events occur only at ~30-min polls -> tau < 0.5h is unidentifiable).

    python -m analysis.estimate_tau                 # seeded coined+lexical units
    python -m analysis.estimate_tau --with-behaviours

Writes out/tau_estimate.csv (both curves) and figures/fig_tau_estimate.pdf.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from analysis.endogenous_hawkes import (
    DAY_H, MIN_EVENTS, BEHAVIOURS, load_marked_corpus, get_units, _pattern,
    meme_events, fit_hawkes,
)

OUT, FIGDIR = "out", "figures"
HEARTBEAT_H = 0.5                       # agent poll cycle -> identifiability floor
LAG_MAX_H = 24.0                        # pairwise-lag window (one background bin)
BIN_H = 0.5                             # non-parametric histogram resolution (= heartbeat)
TAU_GRID = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 9.0, 12.0, 18.0, 24.0])
N_NULL = 20                            # within-day reshuffles for the non-parametric null
NP_EVENT_CAP = 4000                    # subsample denser units for the O(n*k) pair loop


# --------------------------------------------------------------------------- #
# Collect per-unit event arrays (th hours, w marks, days) once.                #
# --------------------------------------------------------------------------- #
def collect_units(corpus, post_text_by_id, t0, with_behaviours=False, seed=0):
    series = []  # (label, th, w, days)
    for label, family, _tier in get_units(smoke=False):
        name = label.split("] ", 1)[-1] if "] " in label else label
        kind = family if family in ("phrase", "hashtag", "domain", "coined") else "emoji"
        th, w, days = meme_events(corpus, _pattern(name, kind), t0,
                                  strict=True, post_text_by_id=post_text_by_id)
        if len(th) >= MIN_EVENTS:
            series.append((label, th, w, days))
    if with_behaviours:
        try:
            from analysis.meme_mining.llm_coded import behaviour_propagate as BP
            emb = BP.embed_corpus(corpus)
            pred, _ = BP.train_predict(emb, corpus, seed=seed)
            cats = [c for c, _ in BP.CATS] if isinstance(BP.CATS[0], tuple) else list(BP.CATS)
            for b in BEHAVIOURS:
                m = pred[:, cats.index(b)].astype(bool)
                th, w, days = meme_events(corpus, None, t0, mask=m)
                if len(th) >= MIN_EVENTS:
                    series.append((f"[behaviour] {b}", th, w, days))
        except Exception as e:  # noqa: BLE001
            print(f"  ! behaviours skipped: {e}", flush=True)
    return series


# --------------------------------------------------------------------------- #
# (A) Non-parametric empirical triggering kernel via within-day reshuffle null. #
# --------------------------------------------------------------------------- #
def _pair_lag_hist(th, edges):
    """Histogram of pairwise lags (t_i - t_j), j<i, with lag <= LAG_MAX_H. Windowed:
    for each i only the earlier events within LAG_MAX_H are paired (searchsorted),
    so cost is O(n * k) with k = events in a 24h window, not O(n^2)."""
    n = len(th)
    h = np.zeros(len(edges) - 1)
    lo_idx = np.searchsorted(th, th - LAG_MAX_H, side="left")   # first j with th_j >= th_i - 24h
    for i in range(1, n):
        j0 = lo_idx[i]
        if j0 < i:
            h += np.histogram(th[i] - th[j0:i], bins=edges)[0]
    return h


def nonparam_kernel(series, rng):
    edges = np.arange(0.0, LAG_MAX_H + BIN_H, BIN_H)
    centers = 0.5 * (edges[:-1] + edges[1:])
    obs = np.zeros(len(centers))
    null = np.zeros(len(centers))
    for _label, th, _w, days in series:
        if len(th) > NP_EVENT_CAP:                    # subsample dense units (sorted-preserving)
            keep = np.sort(rng.choice(len(th), NP_EVENT_CAP, replace=False))
            th, days = th[keep], days[keep]
        T = float(th[-1])
        obs += _pair_lag_hist(th, edges)
        for _ in range(N_NULL):
            # redraw each event uniformly within its own 24h day-bin (clipped to [0,T])
            lo = days * DAY_H
            hi = np.minimum((days + 1) * DAY_H, T)
            hi = np.maximum(hi, lo + 1e-6)
            tp = np.sort(lo + rng.random(len(th)) * (hi - lo))
            null += _pair_lag_hist(tp, edges) / N_NULL
    excess = obs - null
    excess_pos = np.clip(excess, 0, None)
    # empirical half-life: first lag where the (smoothed) excess drops below half its peak
    peak = excess_pos[:6].max() if excess_pos.size else 0.0   # peak within first 3h
    half = np.nan
    if peak > 0:
        below = np.where(excess_pos < 0.5 * peak)[0]
        below = below[centers[below] > BIN_H]                 # skip the 0-bin
        if below.size:
            half = float(centers[below[0]])
    return centers, obs, null, excess, half, peak


# --------------------------------------------------------------------------- #
# (B) Pooled profile likelihood over the tau grid (free per-day background).    #
# --------------------------------------------------------------------------- #
def profile_likelihood(series):
    per_tau_ll = np.zeros(len(TAU_GRID))
    for _label, th, w, days in series:
        for k, tau in enumerate(TAU_GRID):
            per_tau_ll[k] += fit_hawkes(th, w, days, tau=float(tau))["ll"]
    k_hat = int(np.argmax(per_tau_ll))
    tau_hat = float(TAU_GRID[k_hat])
    # 2-log-likelihood-unit interval (approx 95% for 1 dof)
    thresh = per_tau_ll[k_hat] - 2.0
    within = TAU_GRID[per_tau_ll >= thresh]
    return per_tau_ll, tau_hat, (float(within.min()), float(within.max()))


def _plot(centers, excess, per_tau_ll, tau_np, tau_ml, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].bar(centers, excess, width=BIN_H * 0.9, color="#3b6", edgecolor="none")
    ax[0].axvline(HEARTBEAT_H, ls=":", c="k", lw=1, label=f"heartbeat {HEARTBEAT_H}h")
    if np.isfinite(tau_np):
        ax[0].axvline(tau_np, ls="--", c="#c33", lw=1.5, label=f"empirical half-life {tau_np:.2f}h")
    ax[0].set_xlim(0, 12); ax[0].set_xlabel("lag (h)"); ax[0].set_ylabel("excess pairs (obs - null)")
    ax[0].set_title("(A) non-parametric triggering kernel"); ax[0].legend(fontsize=7)
    ax[1].plot(TAU_GRID, per_tau_ll - per_tau_ll.max(), "-o", ms=3, c="#36c")
    ax[1].axhline(-2, ls=":", c="k", lw=1, label="-2 LL")
    ax[1].axvline(tau_ml, ls="--", c="#36c", lw=1.5, label=f"profile MLE {tau_ml:.2g}h")
    ax[1].set_xscale("log"); ax[1].set_xlabel("tau (h)"); ax[1].set_ylabel("pooled profile LL - max")
    ax[1].set_title("(B) profile likelihood"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def _kernel_tau(series, rng):
    """Exponential-kernel tau (h) fitted to the background-controlled excess curve
    for a set of units. Returns (tau, half_life, n_events) or (nan, nan, n)."""
    from scipy.optimize import curve_fit
    centers, obs, null, excess, half, _peak = nonparam_kernel(series, rng)
    n_ev = sum(len(s[1]) for s in series)
    m = (centers > 0) & (centers <= 12) & (excess > 0)
    if m.sum() < 4:
        return float("nan"), half, n_ev
    try:
        p, _ = curve_fit(lambda t, A, tau: A * np.exp(-t / tau), centers[m], excess[m],
                         p0=[excess[m][0], 2.0], maxfev=20000, bounds=([0, 0.1], [np.inf, 48]))
        return float(p[1]), half, n_ev
    except Exception:  # noqa: BLE001
        return float("nan"), half, n_ev


def karma_split(corpus, post_text_by_id, t0, rng, n_bins=3):
    """Test the upvote-dwell hypothesis: does the excitation timescale tau grow with
    a meme's typical karma? (H = recency + 0.2*upvotes, so high-karma posts persist
    longer in the feed -> longer excitation tail.) Estimate the non-parametric tau
    SEPARATELY for unit groups binned by mean upvotes-per-occurrence."""
    units = []
    for label, family, _tier in get_units(smoke=False):
        name = label.split("] ", 1)[-1] if "] " in label else label
        kind = family if family in ("phrase", "hashtag", "domain", "coined") else "emoji"
        th, w, days = meme_events(corpus, _pattern(name, kind), t0,
                                  strict=True, post_text_by_id=post_text_by_id)
        if len(th) >= MIN_EVENTS:
            ubar = float(np.mean((w - 1.0) / 0.2))          # invert w=1+0.2*upvotes -> upvotes
            units.append((label, th, w, days, ubar))
    units.sort(key=lambda u: u[4])
    idx_groups = np.array_split(np.arange(len(units)), n_bins)
    rows = []
    for gi, idx in enumerate(idx_groups):
        g = [units[i] for i in idx]
        if len(g) == 0:
            continue
        ser = [(u[0], u[1], u[2], u[3]) for u in g]
        tau, half, n_ev = _kernel_tau(ser, rng)
        ubars = [u[4] for u in g]
        rows.append({"karma_bin": gi, "n_units": len(g), "n_events": n_ev,
                     "ubar_lo": round(min(ubars), 2), "ubar_hi": round(max(ubars), 2),
                     "ubar_median": round(float(np.median(ubars)), 2),
                     "tau_h": round(tau, 2), "half_life_h": round(half, 2)})
    return pd.DataFrame(rows)


def filter_corpus(corpus, drop_top_k=0, max_cov=None):
    """Cadence filter (Li 2026): remove operator/bot-farm authors before building
    events. drop_top_k: drop the K most prolific authors (concentration/bot-farm cut).
    max_cov: keep only authors whose inter-post CoV <= max_cov (autonomous cut);
    authors with too few posts to fingerprint are KEPT (cannot be classified)."""
    from analysis.cadence_diag import author_cadence
    drop = set()
    n0 = len(corpus)
    if drop_top_k:
        drop |= set(corpus["author_name"].value_counts().head(drop_top_k).index)
    if max_cov is not None:
        cad = author_cadence(corpus)
        drop |= set(cad.loc[cad["cov"] > max_cov, "author"])
    out = corpus[~corpus["author_name"].isin(drop)].reset_index(drop=True)
    print(f"      cadence filter: dropped {len(drop)} authors -> "
          f"{len(out)}/{n0} utterances kept ({100*len(out)/n0:.1f}%)", flush=True)
    return out


def run(with_behaviours=False, seed=0, karma=False, drop_top_k=0, max_cov=None):
    os.makedirs(OUT, exist_ok=True); os.makedirs(FIGDIR, exist_ok=True)
    rng = np.random.default_rng(seed)
    print("[1/3] loading marked corpus ...", flush=True)
    corpus, post_text_by_id = load_marked_corpus(posts_only=True)
    if drop_top_k or max_cov is not None:
        corpus = filter_corpus(corpus, drop_top_k=drop_top_k, max_cov=max_cov)
    t0 = corpus["created_at"].min()
    series = collect_units(corpus, post_text_by_id, t0, with_behaviours, seed)
    n_ev = sum(len(s[1]) for s in series)
    print(f"      {len(series)} units, {n_ev} events total", flush=True)

    print("[2/3] (A) non-parametric empirical kernel ...", flush=True)
    centers, obs, null, excess, tau_np, peak = nonparam_kernel(series, rng)
    frac = float(np.clip(excess, 0, None).sum() / max(obs.sum(), 1))   # share of pairs that are excess
    print(f"      empirical half-life = {tau_np:.2f} h  (excess share {frac:.2f}, peak bin {peak:.0f})",
          flush=True)

    print("[3/3] (B) pooled profile likelihood ...", flush=True)
    per_tau_ll, tau_ml, ci = profile_likelihood(series)
    print(f"      profile-MLE tau_hat = {tau_ml:.2g} h  (2-LL interval {ci[0]:.2g}-{ci[1]:.2g} h)",
          flush=True)
    for tau, ll in zip(TAU_GRID, per_tau_ll):
        print(f"        tau={tau:5.2f}h   pooled LL={ll - per_tau_ll.max():+8.1f}", flush=True)

    pd.DataFrame({"lag_h": centers, "obs": obs, "null": null, "excess": excess}).to_csv(
        os.path.join(OUT, "tau_estimate_kernel.csv"), index=False)
    pd.DataFrame({"tau_h": TAU_GRID, "pooled_ll": per_tau_ll}).to_csv(
        os.path.join(OUT, "tau_estimate_profile.csv"), index=False)
    _plot(centers, excess, per_tau_ll, tau_np, tau_ml,
          os.path.join(FIGDIR, "fig_tau_estimate.pdf"))
    print(f"\nSUMMARY: non-parametric half-life {tau_np:.2f}h | profile-MLE {tau_ml:.2g}h | "
          f"heartbeat floor {HEARTBEAT_H}h", flush=True)

    if karma:
        print("\n[karma] upvote-dwell test: tau by mean-karma group ...", flush=True)
        ks = karma_split(corpus, post_text_by_id, t0, rng)
        ks.to_csv(os.path.join(OUT, "tau_estimate_karma.csv"), index=False)
        print(ks.to_string(index=False), flush=True)
        if ks["tau_h"].notna().sum() >= 2:
            r = np.corrcoef(ks["ubar_median"], ks["tau_h"])[0, 1]
            print(f"[karma] corr(mean-karma, tau) = {r:+.2f}  "
                  f"({'longer tau with karma -> upvote-dwell supported' if r > 0.3 else 'no clear karma-tau link'})",
                  flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-behaviours", action="store_true")
    ap.add_argument("--karma", action="store_true", help="upvote-dwell test: tau by karma group")
    ap.add_argument("--drop-top-k", type=int, default=0, help="drop K most prolific authors (bot-farm cut)")
    ap.add_argument("--max-cov", type=float, default=None, help="keep only authors with CoV<=this (autonomous cut)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(with_behaviours=args.with_behaviours, seed=args.seed, karma=args.karma,
        drop_top_k=args.drop_top_k, max_cov=args.max_cov)
