"""Edge-mediated (follower-graph) expected installs + amplification across seed connectivity.
The edge counterpart of sandbox/expected_installs_surface.py (same table shapes, p5/p95 tails,
non-transmitting-baseline column).

MODEL VERSION: docs/SIMULATION_MODEL.md v2 (pinned P(post)=0.10; audited by
docs/audit_simulation_model.py).

MECHANISM (sandbox.reach_edge.simulate_edge_feed): the payload spreads agent -> follower along
the SNAP ego-Twitter follower graph ($N{=}81{,}306$); each follower's feed is its neighbours'
posts in RANDOM order (no karma), so reach is driven purely by SAR x graph structure. Cascades
are heavy-tailed, so we report the ACTUAL outcome distribution (mean with [p5, p95]).

Axes (the edge analogue of the state model's feed rankings x carrier):
  * "feed ranking" -> SEED CONNECTIVITY prototype (p99/p90/p50/p10 follower-count percentile) --
    the big lever on edge reach.
  * carrier -> there is only ONE carrier on this substrate (security-warning Trojan vs bare
    generic link, same payload); the random-order feed has no upvote channel, so the 10-carrier
    axis does not apply. Hence no by-carrier table (unlike the state model).

Per (model, seed prototype) we draw three outcome distributions:
  * Trojan   : SAR_T = P_POST_REF * P(payload|post, security)         -> installs column
  * Generic  : SAR_G = P_POST_REF * P(payload|post, bare generic)     -> amplification denominator
  * Baseline : SAR = 0 (no retransmission; the seed's direct followers only) -> non-tx baseline
Amplification is expressed relative to the generic link-sharer's EXPECTED installs (mean =
E[inst_T]/E[inst_G], the amplification factor; p5/p95 are the Trojan tail in generic-mean units).

Two tables:
  * tab_expinst_edge_headline : per seed prototype, MEAN over models; installs, non-tx baseline,
    amplification, each with (p5, p95).
  * tab_expinst_edge_bymodel  : robustness by model.

All five models share the same pinned P_POST_REF, so all appear (no reliable-SAR restriction --
that gate is specific to the state model's measured retransmission rate).

    PYTHONPATH=. python -m sandbox.expected_installs_edge --reps 3000

Outputs: out/attack_reach/expected_installs_edge.csv,
figures/tab_expinst_edge_{headline,bymodel}.tex
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from sandbox.reach_edge import (load_follower_graph, _dist, _tx, _sar_decay_k,
                                MODELS, P_POST_REF)
from sandbox.reach_by_model import _pcurves_with_payload
from sandbox.ccdf_trojan_baseline import unified_for, install_rank0

OUT = "out/attack_reach"
FIG = "figures"
PROTOS = [("p99", "p99 (hub)"), ("p90", "p90"), ("p50", "p50 (median)")]  # p10 (1 follower) dropped
M_MIX = 60000


NBOOT = 2000


def _boot(a, seed=0):
    """(mean, lo, hi) with a bootstrap 95% CI of the MEAN over the i.i.d. cascade draws."""
    a = np.asarray(a, float); rng = np.random.default_rng(seed); n = len(a)
    bs = np.array([a[rng.integers(0, n, n)].mean() for _ in range(NBOOT)])
    return (float(a.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def _boot_ratio(t, g, seed=0):
    """(mean, lo, hi) with a bootstrap 95% CI of E[t]/E[g] (independent resamples)."""
    t = np.asarray(t, float); g = np.asarray(g, float); rng = np.random.default_rng(seed)
    m = t.mean() / max(g.mean(), 1e-9)
    bs = np.array([t[rng.integers(0, len(t), len(t))].mean()
                   / max(g[rng.integers(0, len(g), len(g))].mean(), 1e-9) for _ in range(NBOOT)])
    return (float(m), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def _boot_ratio_avg(pairs, seed=0):
    """(mean, lo, hi) of the MEAN OVER MODELS of per-model amplification E[t_k]/E[g_k]."""
    rng = np.random.default_rng(seed)
    pt = float(np.mean([t.mean() / max(g.mean(), 1e-9) for t, g in pairs]))
    bs = np.empty(NBOOT)
    for b in range(NBOOT):
        vals = [t[rng.integers(0, len(t), len(t))].mean() / max(g[rng.integers(0, len(g), len(g))].mean(), 1e-9)
                for t, g in pairs]
        bs[b] = np.mean(vals)
    return (pt, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def _boot_ratio_pool(pairs, seed=0):
    """(mean, lo, hi) of the RATIO OF MEANS across models: (sum_k E[t_k]) / (sum_k E[g_k]).
    Matches the state panel's amplification definition; robust to a baseline near zero for some model."""
    rng = np.random.default_rng(seed)
    pt = float(sum(t.mean() for t, _ in pairs) / max(sum(g.mean() for _, g in pairs), 1e-9))
    bs = np.empty(NBOOT)
    for b in range(NBOOT):
        ts = sum(t[rng.integers(0, len(t), len(t))].mean() for t, _ in pairs)
        gs = sum(g[rng.integers(0, len(g), len(g))].mean() for _, g in pairs)
        bs[b] = ts / max(gs, 1e-9)
    return (pt, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def _fmt(mean, lo, hi, d=0):
    return f"{mean:,.{d}f} [{lo:,.{d}f}, {hi:,.{d}f}]".replace(",", "{,}")


def _pool(arrs):
    """equal-weight pool of per-model draw arrays (down-sample to the smallest so weights are equal)."""
    arrs = [np.asarray(a, float) for a in arrs if a is not None and len(a)]
    if not arrs:
        return None
    return np.concatenate(arrs)


def compute(reps=3000, seed0=0):
    """Run the edge cascade and return a stats bundle for the (combined or standalone) tables:
        dict(N, foll, PROTOS,
             cell={(model, proto): dict(inst=, base=, amp=)},   # each a _stats dict
             headline={proto: dict(inst=, base=, amp=)})        # equal-weight pooled over models
    Also writes the tidy CSV out/attack_reach/expected_installs_edge.csv."""
    rng = np.random.default_rng(seed0)
    print("loading ego-Twitter follower graph...", flush=True)
    N, indptr, in_idx, indeg, outdeg = load_follower_graph()
    pos = indeg[indeg > 0]
    foll = {p: round(float(np.percentile(pos, int(p[1:])))) for p, _ in PROTOS}
    print(f"N={N:,} | edges={in_idx.size:,} | reps={reps} | "
          + ", ".join(f"{p}={foll[p]}" for p, _ in PROTOS), flush=True)

    draws = {}      # (model, proto) -> dict(T=Trojan installs, G=generic installs, B=baseline)
    cell = {}       # (model, proto) -> dict(inst=(m,lo,hi), base=(m,lo,hi), amp=(m,lo,hi))  95% CI
    rows = []
    for name, tag in MODELS:
        pc = _pcurves_with_payload(unified_for(tag))
        ir = install_rank0(tag); inst = ir if ir is not None else 0.66
        sarT = P_POST_REF * _tx(tag, "Trojan"); sarG = P_POST_REF * _tx(tag, "Generic")
        sarM = P_POST_REF * _tx(tag, "market-tip")     # weak control carrier (alpha)
        k, _m = _sar_decay_k(tag)
        for proto, plabel in PROTOS:
            rT, iT = _dist(reps, proto, sarT, inst, indptr, in_idx, outdeg, N, indeg, pc, rng, sar_k=k)
            rG, iG = _dist(reps, proto, sarG, inst, indptr, in_idx, outdeg, N, indeg, pc, rng, sar_k=k)
            rM, iM = _dist(reps, proto, sarM, inst, indptr, in_idx, outdeg, N, indeg, pc, rng, sar_k=k)
            # installs drive the E[installs] columns; reach (exposure) drives amplification, so amp is
            # install-rate-free and defined even for a model that never installs.
            draws[(name, proto)] = dict(T=iT, G=iG, M=iM, Tr=rT, Gr=rG, Mr=rM)
            si, sg, sm = _boot(iT), _boot(iG), _boot(iM)         # installs, generic base, market-tip base
            sa, sam = _boot_ratio(rT, rG), _boot_ratio(rT, rM)   # EXPOSURE amp vs generic / vs market-tip
            cell[(name, proto)] = dict(inst=si, gbase=sg, mbase=sm, amp=sa, amp_mkt=sam)
            rows.append(dict(model=name, seed=proto, followers=foll[proto],
                             inst_mean=si[0], inst_lo=si[1], inst_hi=si[2],
                             gbase_mean=sg[0], gbase_lo=sg[1], gbase_hi=sg[2],
                             mbase_mean=sm[0], mbase_lo=sm[1], mbase_hi=sm[2],
                             amp_mean=sa[0], amp_lo=sa[1], amp_hi=sa[2],
                             amp_mkt_mean=sam[0], amp_mkt_lo=sam[1], amp_mkt_hi=sam[2]))
            print(f"  {name:16s} {proto:3s} foll~{foll[proto]:<4d} "
                  f"inst {si[0]:7.1f}[{si[1]:.0f},{si[2]:.0f}]  gbase {sg[0]:6.1f} mkt {sm[0]:6.1f}  "
                  f"amp/gen {sa[0]:.2f}  amp/mkt {sam[0]:.2f}", flush=True)

    headline = {}
    for proto, _ in PROTOS:
        keys = [(m, proto) for m, _ in MODELS if (m, proto) in draws]
        T = _pool([draws[k]["T"] for k in keys]); G = _pool([draws[k]["G"] for k in keys])
        M = _pool([draws[k]["M"] for k in keys])
        pg = [(draws[k]["Tr"], draws[k]["Gr"]) for k in keys]   # EXPOSURE (reach) ratio pairs
        pm = [(draws[k]["Tr"], draws[k]["Mr"]) for k in keys]
        headline[proto] = dict(inst=_boot(T), gbase=_boot(G), mbase=_boot(M),
                               amp=_boot_ratio_pool(pg), amp_mkt=_boot_ratio_pool(pm))
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "expected_installs_edge.csv"), index=False)
    return dict(N=N, foll=foll, PROTOS=PROTOS, cell=cell, headline=headline, raw=draws)


def build(reps=3000, seed0=0):
    """Standalone edge tables (debug only; the paper uses the merged tables from
    sandbox.expected_installs_surface, which splice in these rows under a section header)."""
    res = compute(reps=reps, seed0=seed0)
    hdr5 = ("E[installs] & generic & market-tip & Amp$_{\\text{gen}}\\times$ & Amp$_{\\text{mkt}}\\times$\\\\")
    for stem, first_col, rows in [("headline", "Seed connectivity", rows_headline(res)),
                                  ("bymodel", "Model & seed", rows_bymodel(res))]:
        ncol = "@{}l r r r r r@{}" if stem == "headline" else "@{}l l r r r r r@{}"
        lines = ["\\begin{table}[t]\\centering\\small", "\\begin{tabular}{" + ncol + "}",
                 "\\toprule", f"{first_col} & {hdr5}", "\\midrule", *rows,
                 "\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
        open(os.path.join(FIG, f"tab_expinst_edge_{stem}.tex"), "w").write("\n".join(lines))
    print(f"\nwrote {OUT}/expected_installs_edge.csv and "
          f"{FIG}/tab_expinst_edge_{{headline,bymodel}}.tex")


def _flab(res, proto):
    """Seed label = percentile code + follower count, e.g. 'p99 (364)' (replaces hub/median/leaf)."""
    return f"\\texttt{{{proto}}} ({res['foll'][proto]:,})".replace(",", "{,}")


def rows_headline(res):
    """Edge headline data rows (6 cells: seed | installs | gbase | market-tip | amp_gen | amp_mkt)."""
    out = []
    for proto, _plabel in res["PROTOS"]:
        h = res["headline"][proto]
        out.append(f"{_flab(res, proto)} & {_fmt(*h['inst'])} & {_fmt(*h['gbase'])} & "
                   f"{_fmt(*h['mbase'])} & {_fmt(*h['amp'], d=2)} & {_fmt(*h['amp_mkt'], d=2)}\\\\")
    return out


def rows_bymodel(res):
    """Edge by-model data rows (7 cells: model | seed | installs | gbase | market-tip | amp_gen | amp_mkt)."""
    out = []
    for name, tag in MODELS:
        first = True
        for proto, _plabel in res["PROTOS"]:
            c = res["cell"].get((name, proto))
            if c is None:
                continue
            mcell = (f"\\texttt{{{name}}}" if first else "")
            out.append(f"{mcell} & {_flab(res, proto)} & "
                       f"{_fmt(*c['inst'])} & {_fmt(*c['gbase'])} & {_fmt(*c['mbase'])} & "
                       f"{_fmt(*c['amp'], d=2)} & {_fmt(*c['amp_mkt'], d=2)}\\\\")
            first = False
        out.append("\\addlinespace")
    return out


def _table_headline(res):
    nstr = f"{res['N']:,}".replace(",", "{,}")
    lines = ["% Edge Table 1 (standalone) -- see the combined tables for the paper version.",
             "\\begin{table}[t]\\centering\\small",
             "\\caption{\\textbf{Edge-mediated expected installs and amplification across seed "
             "connectivity.} Follower-graph retransmission cascade (SNAP ego-Twitter, $N{=}" + nstr
             + "$), per seeded Trojan post; mean over models, $[p_5,p_{95}]$ tails.}",
             "\\label{tab:expinst-edge-headline}"] + tabular_headline(res) + ["\\end{table}", ""]
    open(os.path.join(FIG, "tab_expinst_edge_headline.tex"), "w").write("\n".join(lines))


def _table_bymodel(res):
    lines = ["% Edge Table A (standalone) -- see the combined tables for the paper version.",
             "\\begin{table}[t]\\centering\\small",
             "\\caption{\\textbf{Edge-mediated robustness by model.} Follower-graph cascade by seed "
             "connectivity; installs, non-transmitting baseline, amplification, $[p_5,p_{95}]$ tails.}",
             "\\label{tab:expinst-edge-bymodel}"] + tabular_bymodel(res) + ["\\end{table}", ""]
    open(os.path.join(FIG, "tab_expinst_edge_bymodel.tex"), "w").write("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3000)
    ap.add_argument("--seed0", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    build(reps=args.reps, seed0=args.seed0)
