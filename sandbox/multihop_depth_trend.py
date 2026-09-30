"""Is the change in virality across rephrasing depth a real trend, or noise / a model-specific quirk?

Two nested pseudoreplication traps, both fatal if ignored:
  1. Within a chain, each depth has ONE seed -> the n_author/n_judge trials are pseudo-replicates of that
     one rephrasing. So the CHAIN (not the trial) is the unit, and we work from per-chain slopes.
  2. Chains are CLUSTERED BY MODEL (same model, persona pool, and identical pristine hop-0 seed). Pooling
     chains across models and testing one slope treats within-model chains as independent replicates of a
     universal effect -- anti-conservative for any cross-model claim, and it can be driven entirely by one
     model that paraphrases differently. So the primary analysis is PER MODEL, and an effect is called
     model-general only if it replicates in EVERY model. The pooled test is reported second, clearly
     caveated.

Per model, per metric: per-chain OLS slope of the proportion vs depth; then across that model's chains a
sign test (P(>= observed majority) under a fair-coin null) and, for K>=3, a one-sample t-test with 95% CI.
An effect is MODEL-GENERAL when every model's chains lean the same way (and ideally each is significant).

Caveat baked into the print-out: chains that end early (`chain_end`) do so BECAUSE retransmission failed,
so the retransmit slopes are subject to outcome-dependent truncation; a full-chain-only sensitivity count
is printed. r_up is largely unaffected (judged every depth regardless).

    PYTHONPATH=. python -m sandbox.multihop_depth_trend [--min-depths 2]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np

RUP_DIR = "data/multihop-rup-2026-09"
METRICS = ("retransmit_link", "retransmit_carrier", "r_up_trojan", "r_up_carrier")
NICE = {"retransmit_link": "retransmit link", "retransmit_carrier": "retransmit carrier",
        "r_up_trojan": "r_up trojan", "r_up_carrier": "r_up carrier"}


def _chains():
    """unit-id -> {backend, ndepth_full, metric -> {depth: (k, n)}}. One unit per chain file."""
    units = {}
    for f in sorted(glob.glob(os.path.join(RUP_DIR, "rup_*_x_sec_mg*.json"))):
        d = json.load(open(f))
        tag = d.get("tag", os.path.basename(f))
        uid = f"{tag}#c{d.get('chain', 0)}"
        hs = d.get("hops", [])
        u = units.setdefault(uid, {"backend": tag, "ndepth": len(hs), **{m: {} for m in METRICS}})
        for r in hs:
            h, na = int(r["hop"]), int(r.get("n_author", 30))
            if r.get("p_retransmit_link") is not None:
                u["retransmit_link"][h] = (int(round(r["p_retransmit_link"] * na)), na)
            if r.get("p_retransmit_carrier") is not None:
                u["retransmit_carrier"][h] = (int(round(r["p_retransmit_carrier"] * na)), na)
            for metric, key in (("r_up_trojan", "judge_seed"), ("r_up_carrier", "judge_carrier")):
                tr = [t for t in (r.get(key) or []) if t.get("parsed")]
                if tr:
                    u[metric][h] = (sum(1 for t in tr if t.get("up")), len(tr))
    return units


def _slope(cells):
    if len(cells) < 2:
        return None
    hs = sorted(cells)
    x = np.array(hs, float); y = np.array([cells[h][0] / cells[h][1] for h in hs], float)
    return float(np.polyfit(x - x.mean(), y, 1)[0])


def _sign_p(slopes):
    """Two-sided sign test on the majority direction (fair-coin null)."""
    K = len(slopes); neg = sum(s < 0 for s in slopes); pos = sum(s > 0 for s in slopes)
    maj = max(neg, pos)
    tail = sum(math.comb(K, i) for i in range(maj, K + 1)) / 2 ** K
    return min(1.0, 2 * tail), neg, pos


def _t_ci(slopes):
    a = np.array(slopes, float); K = len(a)
    if K < 3:
        return float(a.mean()), None, None
    from scipy import stats
    m, sd = a.mean(), a.std(ddof=1); se = sd / math.sqrt(K)
    p = 2 * stats.t.sf(abs(m / se), K - 1) if se > 0 else 0.0
    h = stats.t.ppf(0.975, K - 1) * se
    return float(m), float(p), (float(m - h), float(m + h))


def _row(label, slopes):
    m, tp, ci = _t_ci(slopes)
    sp, neg, pos = _sign_p(slopes)
    tp_s = f"{tp:.3f}" if tp is not None else " -- "
    ci_s = f"[{ci[0]:+.3f},{ci[1]:+.3f}]" if ci else "  --  "
    sig = (sp < 0.05) or (tp is not None and tp < 0.05)
    return (f"   {label:<14}K={len(slopes):<3}{f'{neg}-/{pos}+':>7}  mean={m:>+7.3f}  "
            f"sign p={sp:>6.3f}  t p={tp_s:>6}  95%CI={ci_s}{'  *' if sig else ''}"), \
           dict(neg=neg, pos=pos, K=len(slopes), mean=m, sign_p=sp, t_p=tp, sig=sig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-depths", type=int, default=2, help="min depths for a chain to yield a slope")
    a = ap.parse_args()
    units = _chains()
    models = sorted({u["backend"] for u in units.values()})
    # per-model chain slopes
    per = {m: {bk: [] for bk in models} for m in METRICS}
    for uid, d in units.items():
        for metric in METRICS:
            if len(d[metric]) >= a.min_depths:
                s = _slope(d[metric])
                if s is not None:
                    per[metric][d["backend"]].append(s)

    counts = {bk: sum(1 for u in units.values() if u["backend"] == bk) for bk in models}
    full = {bk: sum(1 for u in units.values() if u["backend"] == bk and u["ndepth"] >= 4) for bk in models}
    print("Depth-trend test -- PRIMARY: per model (chain = unit; effect is model-general only if every "
          "model agrees)\n")
    print("chains: " + ",  ".join(f"{bk}={counts[bk]} (full-4-depth={full[bk]})" for bk in models) + "\n")

    MINK = 3                                              # a model must have >=3 chains to weigh in on generality
    for metric in METRICS:
        print(f"[{NICE[metric]}]")
        judged = []                                       # (direction, consistent) for models with K>=MINK
        for bk in models:
            sl = per[metric][bk]
            if not sl:
                print(f"   {bk:<14}(no chains yet)"); continue
            line, st = _row(bk, sl)
            print(line)
            if st["K"] >= MINK:
                d = "-" if st["neg"] > st["pos"] else "+" if st["pos"] > st["neg"] else "0"
                consistent = (st["sign_p"] < 0.10) or (st["t_p"] is not None and st["t_p"] < 0.05)
                judged.append((d, consistent))
        if len(judged) < 2:
            print("   => insufficient models (need >=2 with >=3 chains) to judge generality")
        elif all(d == judged[0][0] and d != "0" and c for d, c in judged):
            print(f"   => MODEL-GENERAL: every model ({len(judged)}) is directionally consistent and leans "
                  f"{judged[0][0]}")
        elif all(d == judged[0][0] and d != "0" for d, c in judged):
            print(f"   => leans {judged[0][0]} in all models but not each is individually consistent "
                  f"(suggestive, not firmly general)")
        else:
            print("   => model-DEPENDENT: models disagree or a model is flat -> do NOT claim as general")
        print()

    print("-" * 90)
    print("SECONDARY (caveated, anti-conservative -- ignores model clustering; do not use for cross-model "
          "claims):")
    for metric in METRICS:
        pooled = [s for bk in models for s in per[metric][bk]]
        if pooled:
            line, _ = _row(NICE[metric], pooled); print(line)
    print("\nNote: retransmit metrics are subject to outcome-dependent truncation (chains end early when "
          "retransmission fails); see full-4-depth counts above. r_up is judged at every depth regardless.")


if __name__ == "__main__":
    main()
