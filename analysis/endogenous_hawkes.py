"""State-mediated (feed/visibility) contagiousness: the PRIMARY Moltbook estimate.

MECHANISM: STATE-MEDIATED (endogenous / platform). This module needs NO interaction
graph — Moltbook contagion is measured state-mediated, never over a comment/reply
graph (the only edge substrate in the project is the Twitter follower graph used by
the edge-mediated reach cascade, sandbox/reach_edge.py). It models each meme as a
univariate marked self-exciting (Hawkes) process whose excitation is weighted by
each occurrence's modeled FEED VISIBILITY under Moltbook's known ranking operator
H = recency + 0.2*upvotes (sandbox/forum.py::_score, sandbox/real_feed.py). The
causal loop is meme -> engagement (upvotes) -> feed visibility -> new occurrences,
i.e. A's occurrence raises B's exposure through a shared platform state, not
through an A->B edge. See README.

Model (per meme m), events = per-author first occurrence at t_j with visibility
mark w_j = 1 + 0.2*upvotes_j (the upvote term of H; recency enters via the kernel):

    lambda_m(s) = mu_m(day(s)) + alpha_m * sum_{t_j < s} w_j * exp(-(s - t_j)/tau)

  - mu_m: piecewise-constant background (per 24h bin) absorbs common shocks /
    platform growth / exogenous fads.
  - tau: excitation timescale, MEASURED at 2.5h (analysis/estimate_tau.py: exponential
    fit to the background-controlled non-parametric triggering kernel).
  - Fit (mu_m, alpha_m) by exponential-kernel Hawkes MLE (Ogata recursion, O(N)).

Reported per meme:
  - R_endo = alpha * tau * mean(w)  : endogenous BRANCHING RATIO (offspring per
    occurrence). Criticality is R_endo > 1. Scale-invariant to how w is normalized.
  - eta    = integral(lambda_endo) / integral(lambda) over the window : realized
    endogenous mass fraction in [0,1).
  - Visibility test (does feed visibility drive excitation?): log-likelihood gain
    of the marked model over (a) an unweighted w=1 model and (b) upvote-shuffled
    nulls, with a permutation p-value. If the marked fit is no better than the
    shuffle, R_endo is not visibility-mediated.

    python -m analysis.endogenous_hawkes            # full
    python -m analysis.endogenous_hawkes --smoke     # a few units, quick

Outputs: out/endogenous_contagion.csv, figures/fig_endogenous_Rendo.pdf.
"""

from __future__ import annotations

import argparse
import zlib
import os
import re

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

from analysis import load

OUT, FIGDIR = "out", "figures"
TAU_H = 2.5          # excitation timescale (h), MEASURED: exponential fit to the background-controlled
#                      non-parametric triggering kernel = 2.42h (analysis/estimate_tau.py). A few agent
#                      heartbeats (~30min floor); NOT the single heartbeat, NOT the 20.3h feed half-life.
DAY_H = 24.0         # background bin width (hours)
KARMA_W = 0.2        # upvote weight in H (recency + 0.2*upvotes)
N_SHUFFLE = 20       # upvote-shuffle placebo repetitions
PHI_NULL_B = 199     # inter-event-gap permutations for the per-unit phi (directionality) null
MIN_EVENTS = 40      # skip memes with fewer per-author first occurrences

# Topic keyword-proxies (regex) and LLM-coded behaviours — for parity with the
# edge-mediated overview table (all four meme families). Topic regexes verbatim
# from regen_overview_feedseeding.py; behaviours via analysis.behaviour_propagate.
# Topics are now data-driven BERTopic clusters (analysis.topic_clusters), reduced
# to TOPIC_K and assigned over the FULL corpus — replacing the old 6 hand-written
# keyword-proxy regexes, which were too coarse.
TOPIC_K = 40
BEHAVIOURS = ["security_warning", "money_making", "self_promotion",
              "sharing_resource", "introducing_self", "social_support"]


# --------------------------------------------------------------------------- #
# Marked corpus: one row per authored utterance, carrying its upvotes.         #
# load.corpus() drops upvotes, so we rebuild locally from posts + comments     #
# (both parquet files carry `upvotes`). Same author/text/time columns so the   #
# regex detectors behave identically to the rest of the pipeline.              #
# --------------------------------------------------------------------------- #
def load_marked_corpus(posts_only: bool = True, cluster_npy: str | None = None
                       ) -> tuple[pd.DataFrame, dict]:
    """Returns (corpus, post_text_by_id). `carrier_post_id` on each row identifies
    the post the utterance lives on (a post's own id, or a comment's parent post),
    used by the strict occurrence rule to exclude comments on a post that already
    carries the meme (localized discussion, not re-transmission).

    posts_only (default True): restrict occurrences to POSTS. The feed ranks posts,
    so state-mediated (feed/visibility) contagion is a post-level phenomenon; the
    corpus is ~86% reply comments (chatter) that never enter the ranked feed. With
    posts_only the strict/inclusive distinction is moot (every occurrence is a post)."""
    posts = load.load_posts()
    p = posts[["author_name", "text", "created_at", "upvotes", "submolt_name", "id"]].copy()
    p["source"] = "post"
    p = p.rename(columns={"id": "carrier_post_id"})
    if cluster_npy is not None:                    # data-driven cluster id per post (aligned to load_posts)
        ca = np.load(cluster_npy)
        assert len(ca) == len(p), f"cluster assignment {len(ca)} != posts {len(p)}"
        p["cluster"] = ca
    if posts_only:
        out = p
    else:
        comments = load.load_comments()
        cm = comments[["author_name", "text", "created_at", "upvotes", "post_id"]].copy()
        cm["submolt_name"] = pd.NA
        cm["source"] = "comment"
        cm = cm.rename(columns={"post_id": "carrier_post_id"})
        out = pd.concat([p, cm], ignore_index=True)
    out = out.dropna(subset=["author_name", "created_at"])
    out = out[out["text"].str.len() > 0].reset_index(drop=True)
    out["upvotes"] = out["upvotes"].fillna(0).clip(lower=0)
    post_text_by_id = dict(zip(posts["id"], posts["text"].fillna("")))
    return out, post_text_by_id


def _pattern(meme: str, kind: str) -> re.Pattern:
    """Regex for a meme string. Mirrors analysis.run._meme_to_pattern, plus a
    word-boundary form for single-token coinages."""
    if kind == "phrase" or " " in meme:
        return re.compile(r"\b" + r"[\W_]+".join(re.escape(w) for w in meme.split()) + r"\b",
                          re.IGNORECASE)
    if kind in ("hashtag", "domain"):
        return re.compile(re.escape(meme), re.IGNORECASE)
    if kind == "coined":
        if re.fullmatch(r"\w+", meme):                       # single alnum token
            return re.compile(r"\b" + re.escape(meme) + r"\b", re.IGNORECASE)
        return re.compile(re.escape(meme), re.IGNORECASE)
    return re.compile(re.escape(meme))                       # emoji


def meme_events(corpus: pd.DataFrame, pattern, t0: pd.Timestamp,
                strict=False, post_text_by_id: dict | None = None, mask=None):
    """Per-author FIRST occurrence of the meme. Returns (times_h, marks, days)
    sorted by time — times in hours since t0, marks w_j = 1 + 0.2*upvotes_j,
    days = 24h-bin index. Empty arrays if too rare.

    strict=True: count an occurrence only when it is a NEW POST carrying the meme,
    or a COMMENT introducing it on a post that does not already carry it. Comments
    on a post that already carries the meme (localized thread discussion) are
    excluded — the state-mediated analogue of the edge-table's seed-reply exclusion.
    `mask` (a boolean selector aligned to corpus) overrides `pattern` — for behaviours."""
    if mask is None:
        mask = corpus["text"].str.contains(pattern, regex=True, na=False)
    else:
        mask = pd.Series(np.asarray(mask, dtype=bool), index=corpus.index)
    cols = ["author_name", "created_at", "upvotes", "source", "carrier_post_id"]
    hits = corpus.loc[mask, cols]
    if hits.empty:
        return np.array([]), np.array([]), np.array([])
    if strict and post_text_by_id is not None:
        is_post = hits["source"].to_numpy() == "post"
        parent_text = hits["carrier_post_id"].map(post_text_by_id).fillna("")
        parent_carries = parent_text.str.contains(pattern, regex=True, na=False).to_numpy()
        hits = hits[is_post | ~parent_carries]
        if hits.empty:
            return np.array([]), np.array([]), np.array([])
    first = (hits.sort_values("created_at")
                 .groupby("author_name", sort=False)
                 .first()
                 .reset_index())
    th = (first["created_at"] - t0).dt.total_seconds().to_numpy() / 3600.0
    order = np.argsort(th, kind="stable")
    th = th[order]
    w = (1.0 + KARMA_W * first["upvotes"].to_numpy(dtype="float64")[order])
    days = np.floor(th / DAY_H).astype(int)
    return th, w, days


# --------------------------------------------------------------------------- #
# Exponential-kernel marked Hawkes MLE (Ogata O(N) recursion, per-day mu).      #
# --------------------------------------------------------------------------- #
def _ogata_R(th, w, tau):
    """R_i = sum_{j<i} w_j exp(-(t_i - t_j)/tau). Depends only on (th, w, tau) —
    NOT on the fitted params — so compute once per fit, not per likelihood eval."""
    n = len(th)
    R = np.empty(n)
    R[0] = 0.0
    for i in range(1, n):
        R[i] = np.exp(-(th[i] - th[i - 1]) / tau) * (R[i - 1] + w[i - 1])
    return R


def _neg_ll(theta, R, days, day_len, Sconst, n_days):
    """theta = [log mu_0 .. log mu_{D-1}, log alpha]. Vectorized negative LL
    given the precomputed excitation R and compensator constant Sconst."""
    mu = np.exp(theta[:n_days])
    alpha = np.exp(theta[n_days])
    lam = mu[days] + alpha * R
    log_term = np.sum(np.log(np.clip(lam, 1e-12, None)))
    comp = np.sum(mu * day_len) + alpha * Sconst
    return -(log_term - comp)


def _neg_ll_detrend(theta, R, gd, Gsum, Sconst):
    """theta = [log mu0, log alpha]. DETRENDED background: mu(t) = mu0 * g_{d(t)}, where g is a
    FIXED exogenous shape (platform daily activity). Only the scale mu0 is free, so a meme that
    merely tracks platform-wide growth is absorbed by the background and cannot inflate alpha."""
    mu0 = np.exp(theta[0]); alpha = np.exp(theta[1])
    lam = mu0 * gd + alpha * R
    log_term = np.sum(np.log(np.clip(lam, 1e-12, None)))
    comp = mu0 * Gsum + alpha * Sconst
    return -(log_term - comp)


def fit_hawkes(th, w, days, tau=TAU_H, bg_shape=None):
    """Fit alpha and the background. bg_shape=None: free per-day mu_d (original). bg_shape given:
    DETRENDED background mu(t)=mu0*g_{d(t)} with g the fixed platform-activity shape (one scale mu0)
    -- excitation then captures only clustering BEYOND the platform trend."""
    T = float(th[-1]) if len(th) else 0.0
    n_days = int(np.floor(T / DAY_H)) + 1 if T > 0 else 1
    days = np.clip(days, 0, n_days - 1)
    day_len = np.array([min((d + 1) * DAY_H, T) - d * DAY_H for d in range(n_days)])
    day_len = np.clip(day_len, 1e-6, None)

    R = _ogata_R(th, w, tau)                                  # once
    Sconst = float(np.sum(w * tau * (1.0 - np.exp(-(T - th) / tau))))

    # MULTI-START over the alpha initialization: the joint (mu, alpha) likelihood is
    # non-convex and L-BFGS-B can converge to a spurious high-alpha local optimum for
    # dense units at some tau (worse LL than neighbours). Try several alpha seeds and
    # keep the best (lowest neg-LL) -> stable fits across tau.
    ALPHA0 = (1e-3, 1e-2, 1e-1, 0.5)
    if bg_shape is None:
        counts = np.bincount(days, minlength=n_days).astype("float64")
        mu0 = np.clip(counts / day_len, 1e-4, None)
        bounds = [(-25.0, 15.0)] * (n_days + 1)   # log-space; keeps exp() finite
        res = None
        for a0 in ALPHA0:
            theta0 = np.concatenate([np.log(mu0), [np.log(a0)]])
            r = minimize(_neg_ll, theta0, args=(R, days, day_len, Sconst, n_days),
                         method="L-BFGS-B", bounds=bounds, options={"maxiter": 500})
            if res is None or r.fun < res.fun:
                res = r
        theta = res.x
        mu = np.exp(theta[:n_days])
        alpha = float(np.exp(theta[n_days]))
        comp_bg = float(np.sum(mu * day_len))
        # restricted (alpha=0) inhomogeneous-Poisson MLE log-lik: mu_d = counts_d/day_len_d
        mu_p = np.clip(counts / day_len, 1e-12, None)
        ll_bg = float(np.sum(counts * np.log(mu_p)) - np.sum(counts))
    else:
        g = np.asarray(bg_shape, dtype="float64")
        if len(g) < n_days:                                  # align to this fit's window
            g = np.concatenate([g, np.full(n_days - len(g), g.mean() if len(g) else 1.0)])
        g = np.clip(g[:n_days], 1e-6, None)
        Gsum = float(np.sum(g * day_len))
        mu00 = np.log(max(len(th) / max(Gsum, 1e-9), 1e-4))
        res = None
        for a0 in ALPHA0:
            theta0 = np.array([mu00, np.log(a0)])
            r = minimize(_neg_ll_detrend, theta0, args=(R, g[days], Gsum, Sconst),
                         method="L-BFGS-B", bounds=[(-25.0, 15.0), (-25.0, 15.0)],
                         options={"maxiter": 500})
            if res is None or r.fun < res.fun:
                res = r
        theta = res.x
        mu0 = float(np.exp(theta[0]))
        alpha = float(np.exp(theta[1]))
        mu = mu0 * g
        comp_bg = mu0 * Gsum
        # restricted (alpha=0) MLE log-lik under the fixed background shape g
        mu0_p = max(len(th) / max(Gsum, 1e-9), 1e-12)
        ll_bg = float(len(th) * np.log(mu0_p) + np.sum(np.log(g[days])) - len(th))
    ll = -float(res.fun)

    wbar = float(np.mean(w))
    r_endo = alpha * tau * wbar
    comp_ex = alpha * Sconst
    eta = comp_ex / (comp_ex + comp_bg) if (comp_ex + comp_bg) > 0 else float("nan")
    return {"alpha": alpha, "mu": mu, "ll": ll, "ll_bg": ll_bg, "R_endo": r_endo, "eta": eta,
            "wbar": wbar, "tau": tau, "n_events": len(th), "T": T}


def _fit_ll(th, w, days, tau=TAU_H):
    return fit_hawkes(th, w, days, tau)


def _phi_null(th, w, tau, fwd_bg, rev_bg, rng, B):
    """Per-unit null for phi_endo = R_endo_forward - R_endo_reversed.

    Permuting the inter-event gaps (with their attached marks) preserves the multiset of gaps and the
    marks -- hence the overall clustering magnitude and total span T (sum of gaps is permutation-
    invariant) -- but destroys the specific temporal ORDER, so forward and reversed fits become
    exchangeable and phi_null is centred at 0. The observed phi is directionally significant only if it
    exceeds this null. Returns (mean, sd, one-sided p = P(phi_null >= phi_obs), z-score); T is invariant
    so the fixed background shapes are reused unchanged."""
    n = len(th)
    gaps = np.diff(th, prepend=0.0)
    phis = np.empty(B)
    for b in range(B):
        perm = rng.permutation(n)
        th2 = np.cumsum(gaps[perm]); w2 = w[perm]; T2 = th2[-1]
        d2 = np.floor(th2 / DAY_H).astype(int)
        ff = fit_hawkes(th2, w2, d2, tau, bg_shape=fwd_bg)
        thr = (T2 - th2)[::-1].copy(); wr = w2[::-1].copy(); dr = np.floor(thr / DAY_H).astype(int)
        rf = fit_hawkes(thr, wr, dr, tau, bg_shape=rev_bg)
        phis[b] = ff["R_endo"] - rf["R_endo"]
    return phis


def analyse_meme(label, family, tier, corpus, t0, rng, n_shuffle=N_SHUFFLE, tau=TAU_H,
                 strict=False, post_text_by_id=None, pattern=None, mask=None, bg_shape=None,
                 n_phi=0):
    if mask is None and pattern is None:
        pattern = _pattern(label.split("] ", 1)[-1] if "] " in label else label,
                           family if family in ("phrase", "hashtag", "domain", "coined") else "emoji")
    th, w, days = meme_events(corpus, pattern, t0, strict=strict,
                              post_text_by_id=post_text_by_id, mask=mask)
    if len(th) < MIN_EVENTS:
        return None
    # detrended background: restrict the platform shape to this meme's window; reverse it for the placebo
    T = th[-1]
    n_days_fwd = int(np.floor(T / DAY_H)) + 1 if T > 0 else 1
    fwd_bg = np.asarray(bg_shape)[:n_days_fwd] if bg_shape is not None else None
    rev_bg = fwd_bg[::-1].copy() if fwd_bg is not None else None

    marked = fit_hawkes(th, w, days, tau, bg_shape=fwd_bg)
    unw = fit_hawkes(th, np.ones_like(w), days, tau, bg_shape=fwd_bg)  # w=1 null (no visibility mark)

    # Self-excitation significance: likelihood-ratio test of the Hawkes vs its own
    # (alpha=0) inhomogeneous-Poisson background. LR = 2(ll_full - ll_bg); because
    # alpha=0 is on the boundary, the null is a 50:50 chi-bar-squared mix of chi2_0
    # and chi2_1, so p = 0.5*P(chi2_1 > LR). Low p = clustering beyond the per-day
    # background (burstiness/self-excitation), NOT proof of contagion vs common cause.
    lr_excite = max(0.0, 2.0 * (marked["ll"] - marked["ll_bg"]))
    p_excite = 0.5 * float(chi2.sf(lr_excite, 1)) if lr_excite > 0 else 1.0

    # Time-reversal placebo (the phi analogue for a self-exciting process):
    # genuine contagion is FORWARD-directional (past occurrences excite future
    # ones); ambient / homophily / common-shock clustering is time-symmetric.
    # phi_endo = R_endo_forward - R_endo_reversed  (>0 = transmission-consistent, ~0 = ambient).
    # (R_endo is the reported branching ratio; for a subcritical stationary Hawkes it equals the
    # endogenous fraction eta, and the two are >0.99 correlated here -- we use R for consistency.)
    th_rev = (T - th)[::-1].copy()
    w_rev = w[::-1].copy()
    days_rev = np.floor(th_rev / DAY_H).astype(int)
    rev = fit_hawkes(th_rev, w_rev, days_rev, tau, bg_shape=rev_bg)
    phi_endo = marked["R_endo"] - rev["R_endo"]

    # Per-unit directionality null for phi (see _phi_null). One-sided p tests forward transmission.
    phi_null_mean = phi_null_sd = z_phi = float("nan")
    p_phi = float("nan")
    if n_phi and len(th) >= MIN_EVENTS:
        # dedicated per-unit generator (stable across runs, independent of the main rng stream so the
        # upvote-shuffle draws / p_perm stay byte-identical whether or not the phi null is computed)
        phi_rng = np.random.default_rng(zlib.crc32(label.encode("utf-8")))
        phis = _phi_null(th, w, tau, fwd_bg, rev_bg, phi_rng, n_phi)
        phi_null_mean = float(np.mean(phis)); phi_null_sd = float(np.std(phis, ddof=1))
        p_phi = float((np.sum(phis >= phi_endo) + 1) / (n_phi + 1))
        z_phi = (phi_endo - phi_null_mean) / phi_null_sd if phi_null_sd > 0 else float("nan")

    # Upvote-shuffle placebo: permute marks across this meme's events, refit.
    sh_R, sh_ll = [], []
    for _ in range(n_shuffle):
        wp = rng.permutation(w)
        f = fit_hawkes(th, wp, days, tau, bg_shape=fwd_bg)
        sh_R.append(f["R_endo"]); sh_ll.append(f["ll"])
    sh_R = np.array(sh_R); sh_ll = np.array(sh_ll)
    # permutation p: fraction of shuffles fitting at least as well as the true marks
    p_perm = float((np.sum(sh_ll >= marked["ll"]) + 1) / (n_shuffle + 1))

    return {
        "unit": label, "family": family, "tier": tier,
        "n_events": marked["n_events"],
        "R_endo": round(marked["R_endo"], 3),
        "R_endo_rev": round(rev["R_endo"], 3),
        "eta": round(marked["eta"], 3),
        "eta_rev": round(rev["eta"], 3),
        "phi_endo": round(phi_endo, 3),
        "phi_null_mean": round(phi_null_mean, 4),
        "phi_null_sd": round(phi_null_sd, 4),
        "p_phi": round(p_phi, 4) if p_phi == p_phi else float("nan"),
        "z_phi": round(z_phi, 2) if z_phi == z_phi else float("nan"),
        "lr_excite": round(lr_excite, 1),
        "p_excite": p_excite,
        "alpha": marked["alpha"],
        "tau": tau,
        "wbar": round(marked["wbar"], 2),
        "R_endo_unweighted": round(unw["R_endo"], 3),
        "R_endo_shuffle": round(float(np.mean(sh_R)), 3),
        "ll_marked": round(marked["ll"], 1),
        "ll_unweighted": round(unw["ll"], 1),
        "ll_shuffle_mean": round(float(np.mean(sh_ll)), 1),
        "visibility_gain": round(marked["ll"] - unw["ll"], 1),   # marked vs w=1
        "shuffle_gain": round(marked["ll"] - float(np.mean(sh_ll)), 1),
        "p_perm": p_perm,
        "supercritical": bool(marked["R_endo"] > 1.0),
    }


# --------------------------------------------------------------------------- #
# Unit selection: coined (highest-confidence copy-fidelity) + lexical anchors. #
# Behaviours/topics (per-utterance labels) can be added via meme_events on a   #
# precomputed mask; scoped out of the first pass (they need embed/label runs). #
# --------------------------------------------------------------------------- #
def get_units(smoke=False):
    units = []  # (label, family, tier)
    # Coined / neologisms — copy-fidelity tier.
    cc = os.path.join(OUT, "coined_categorized.csv")
    if os.path.exists(cc):
        df = pd.read_csv(cc)
        for _, r in df.iterrows():
            units.append((str(r["meme"]), "coined", "coined"))
    # Lexical anchors — top spread per kind from the meme miner.
    mm = os.path.join(OUT, "memes.csv")
    if os.path.exists(mm):
        m = pd.read_csv(mm)
        take = {"emoji": 6, "hashtag": 6, "domain": 6, "phrase": 12}
        for kind, k in take.items():
            sub = (m[m["kind"] == kind]
                   .sort_values("distinct_authors", ascending=False)
                   .head(k))
            tier = "phrase" if kind == "phrase" else "lexical"
            for _, r in sub.iterrows():
                units.append((f"[{kind}] {r['meme']}", kind, tier))
    # de-dup by label, keep order
    seen, uniq = set(), []
    for u in units:
        if u[0] in seen:
            continue
        seen.add(u[0]); uniq.append(u)
    if smoke:
        uniq = uniq[:8]
    return uniq


def run(smoke=False, seed=0, strict=False, detrend=False, tau=TAU_H, posts_only=True,
        cluster_mcs=200, n_phi=None):
    n_phi = (4 if smoke else PHI_NULL_B) if n_phi is None else n_phi
    os.makedirs(OUT, exist_ok=True); os.makedirs(FIGDIR, exist_ok=True)
    rng = np.random.default_rng(seed)
    mode = "POSTS-ONLY" if posts_only else "posts+comments"
    mode += " | STRICT" if strict else " | inclusive"
    mode += " | DETRENDED background" if detrend else ""
    # data-driven semantic family: bge-large + HDBSCAN clusters (analysis.meme_mining.cluster_v2)
    cluster_npy, cluster_labels = None, None
    if posts_only:
        cnpy = os.path.join("out/cluster_v2", f"assignment_mcs{cluster_mcs}.npy")
        clab = os.path.join("out/cluster_v2", f"labels_mcs{cluster_mcs}.csv")
        if os.path.exists(cnpy) and os.path.exists(clab):
            cluster_npy, cluster_labels = cnpy, pd.read_csv(clab)
    print(f"[1/4] loading marked corpus ... [{mode}]", flush=True)
    corpus, post_text_by_id = load_marked_corpus(posts_only=posts_only, cluster_npy=cluster_npy)
    t0 = corpus["created_at"].min()

    # Platform daily-activity shape g_d (ALL utterances) -> the exogenous background for detrending:
    # a meme that merely rides platform-wide growth is absorbed by mu(t)=mu0*g_d, not by excitation.
    bg = None
    if detrend:
        cdays = np.floor((corpus["created_at"] - t0).dt.total_seconds().to_numpy()
                         / (3600.0 * DAY_H)).astype(int)
        g = np.bincount(cdays[cdays >= 0], minlength=int(cdays.max()) + 1).astype("float64")
        bg = g / max(g.mean(), 1e-9)                      # normalise to mean 1 (a shape)
        print(f"      detrend: platform shape over {len(bg)} days (mean-normalised)", flush=True)

    units = get_units(smoke)
    print(f"[2/4] fitting state-mediated Hawkes for {len(units)} units ...", flush=True)
    rows = []
    for label, family, tier in units:
        try:
            r = analyse_meme(label, family, tier, corpus, t0, rng,
                             n_shuffle=(4 if smoke else N_SHUFFLE), tau=tau,
                             strict=strict, post_text_by_id=post_text_by_id, bg_shape=bg, n_phi=n_phi)
        except Exception as e:  # noqa: BLE001 — keep the sweep alive
            print(f"      ! {label}: {e}", flush=True); continue
        if r is None:
            continue
        rows.append(r)
        print(f"      {label[:34]:34s} R_endo={r['R_endo']:.2f} eta={r['eta']:.2f} "
              f"phi_endo={r['phi_endo']:+.2f} vis+={r['visibility_gain']:.0f}", flush=True)

    # Topics (keyword-proxy regex) and behaviours (LLM classifier) — semantic units,
    # inclusive occurrences (re-transmission is not verbatim). Parity with the edge table.
    if not smoke and posts_only and cluster_labels is not None:
        print(f"[2b] data-driven clusters (bge-large + HDBSCAN, mcs={cluster_mcs}, "
              f"{len(cluster_labels)} clusters) ...", flush=True)
        cl = corpus["cluster"].to_numpy()
        for _, crow in cluster_labels.iterrows():
            cid = int(crow["cluster"]); name = str(crow["label"])
            r = analyse_meme(f"[cluster] {name}", "cluster", "cluster", corpus, t0, rng,
                             tau=tau, strict=False, mask=(cl == cid), bg_shape=bg, n_phi=n_phi)
            if r:
                rows.append(r)
                print(f"      c{cid:3d} {name[:38]:38s} R_endo={r['R_endo']:.2f} "
                      f"phi={r['phi_endo']:+.2f}", flush=True)
    elif not smoke and posts_only:
        print("[2b] no cluster_v2 assignment found; run analysis.meme_mining.cluster_v2 "
              "first for the data-driven semantic family.", flush=True)
    if not smoke and not posts_only:
        print(f"[2b] topics (BERTopic clusters, k={TOPIC_K}, full-corpus assignment) ...", flush=True)
        try:
            from analysis.meme_mining.bertopic import topic_clusters as TC
            tlabels, tassign = TC.load_topics(corpus, k=TOPIC_K)
            assert len(tassign) == len(corpus), f"topic assign {len(tassign)} != {len(corpus)}"
            for tid, name in tlabels:
                m = (tassign == tid)
                r = analyse_meme(f"[topic] {name}", "topic", "topic", corpus, t0, rng,
                                 tau=tau, strict=False, mask=m, bg_shape=bg, n_phi=n_phi)
                if r:
                    rows.append(r)
                    print(f"      {str(name)[:34]:34s} R_endo={r['R_endo']:.2f} phi_endo={r['phi_endo']:+.2f}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"      ! topics skipped: {e}", flush=True)
        print("[2c] behaviours (LLM classifier) ...", flush=True)
        try:
            from analysis.meme_mining.llm_coded import behaviour_propagate as BP
            emb = BP.embed_corpus(corpus)
            assert emb.shape[0] == len(corpus), f"emb {emb.shape[0]} != corpus {len(corpus)}"
            pred, _ = BP.train_predict(emb, corpus, seed=seed)
            cats = [c for c, _ in BP.CATS] if isinstance(BP.CATS[0], tuple) else list(BP.CATS)
            for b in BEHAVIOURS:
                m = pred[:, cats.index(b)].astype(bool)
                r = analyse_meme(f"[behaviour] {b}", "behaviour", "behaviour", corpus, t0, rng,
                                 tau=tau, strict=False, mask=m, bg_shape=bg, n_phi=n_phi)
                if r:
                    rows.append(r)
                    print(f"      {b:24s} R_endo={r['R_endo']:.2f} phi_endo={r['phi_endo']:+.2f}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"      ! behaviours skipped: {e}", flush=True)

    df = pd.DataFrame(rows).sort_values("R_endo", ascending=False).reset_index(drop=True)
    suffix = "_strict" if strict else ""
    path = os.path.join(OUT, f"endogenous_contagion{suffix}.csv")
    df.to_csv(path, index=False)
    print(f"[3/4] wrote {path} ({len(df)} units)", flush=True)

    print("[4/4] figure ...", flush=True)
    _plot(df, os.path.join(FIGDIR, f"fig_endogenous_Rendo{suffix}.pdf"))
    print("\nDone.")
    if not df.empty:
        cols = ["unit", "tier", "n_events", "R_endo", "eta", "eta_rev",
                "phi_endo", "visibility_gain"]
        print(df[cols].head(30).to_string(index=False))


def _plot(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if df.empty:
        return
    d = df.sort_values("R_endo").tail(25)
    colors = {"coined": "#8c2d04", "phrase": "#2a6f97", "lexical": "#5a5a5a"}
    fig, ax = plt.subplots(figsize=(8, max(4, 0.34 * len(d))))
    ax.barh(range(len(d)), d["R_endo"],
            color=[colors.get(t, "#888") for t in d["tier"]])
    ax.axvline(1.0, color="crimson", lw=1.2, ls="--", label="criticality  $R_{endo}=1$")
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([u[:32] for u in d["unit"]], fontsize=7)
    ax.set_xlabel("$R_{endo}$  (state-mediated branching ratio)")
    ax.set_title("State-mediated contagiousness (feed-visibility Hawkes)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax.legend(handles + [ax.lines[0]], list(colors) + ["$R_{endo}=1$"], fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strict", action="store_true",
                    help="exclude comments on posts already carrying the meme (re-transmission only)")
    ap.add_argument("--detrend", action="store_true",
                    help="pin the background to the platform daily-activity shape mu(t)=mu0*g_d, so "
                         "excitation only captures clustering beyond platform-wide growth")
    ap.add_argument("--tau", type=float, default=TAU_H, help="excitation timescale (h); heartbeat ~0.5")
    ap.add_argument("--with-comments", action="store_true",
                    help="include comments (default is POSTS-ONLY: the feed ranks posts)")
    ap.add_argument("--n-phi", type=int, default=None,
                    help=f"gap-permutation reps for the per-unit phi null (default {PHI_NULL_B}; 0 disables)")
    args = ap.parse_args()
    run(smoke=args.smoke, seed=args.seed, strict=args.strict, detrend=args.detrend, tau=args.tau,
        posts_only=not args.with_comments, n_phi=args.n_phi)
