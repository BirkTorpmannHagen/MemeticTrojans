"""Edge-mediated (follower-graph) retransmission cascade — memetic Trojan vs bare generic link.

The state-mediated feed (simulate_endo) makes broadcast a rare karma-gated event and decouples reach
from retransmission. This EDGE-mediated cascade puts retransmission at the center: the payload spreads
agent -> follower -> follower along the SNAP ego-Twitter follower graph, and each follower's feed is
its neighbours' posts in RANDOM order (no karma), so the upvote/climb channel drops out and reach is
driven purely by SAR x graph structure.

Exposure per contact (random-rank neighbour feed, ported from attack_reach_sim.simulate_local_feed
with score-ranking replaced by uniform-random rank):
  * a candidate follower w has feed size F = 1 + Poisson(outdeg[w] * post_rate * window) — the
    ambient neighbour posts plus the payload; the payload sits at a UNIFORM-RANDOM rank r in [0,F).
  * it SEES the payload w.p. spread_any(r) (0 if r >= K_FEED); of those who see, install w.p. the
    per-model rank-0 ASR and retransmit w.p. SAR (-> next frontier). Hubs' large feeds bury the
    payload (degree-dependent dilution).

Reuses the Moltbook action set / rank curves (no data re-collection). Arms differ only in SAR
(= P_POST_REF * P(payload|post), Trojan vs bare generic). Amplification = Trojan / generic.

    python -m sandbox.reach_edge --reps 1000

Outputs: out/attack_reach/reach_raw_edge.csv (model,rep,reach,inst), reach_by_model_edge.csv,
out/attack_reach/amplification_edge.csv, figures/tab_amplification_edge.tex,
figures/reach_edge_mediated.pdf.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from sandbox.attack_reach_sim import curve, K_FEED, CYCLE_H, WINDOW_H, pick_seed
from sandbox.reach_by_model import _pcurves_with_payload
from sandbox.ccdf_trojan_baseline import unified_for, install_rank0

OUT = "out/attack_reach"
FIGDIR = "figures"
FIG = "figures"
TWITTER = "data_ext/twitter_combined.txt.gz"
P_POST_REF = 0.058                         # retransmission posting propensity (mean empirical P(post));
#                                            enters ONLY the SAR = P_POST_REF * P(payload|post).
POST_RATE = 0.013                          # ambient neighbour posting rate (corpus) -> sets each
#                                            follower's FEED SIZE / how buried the payload is. This is a
#                                            DISTINCT quantity from P_POST_REF (feed density vs
#                                            retransmission); do not tie them together.
SAR_DECAY_K_FALLBACK = 0.134              # gpt-oss measured retransmission rank-decay (25x over K feed)
MODELS = [("gpt-oss-120B", "gptoss120b"), ("deepseek-v4-flash", "deepseekflash"),
          ("qwen-32B", "qwen32b"), ("gemma2-27B", "gemma2_27b"), ("command-r-35B", "commandr_35b")]
COL = {"gpt-oss-120B": "#4c78a8", "deepseek-v4-flash": "#e45756", "qwen-32B": "#72b7b2",
       "gemma2-27B": "#eeca3b", "command-r-35B": "#b279a2"}


def load_follower_graph(path=TWITTER):
    """SNAP ego-Twitter follower graph. Edge 'a b' = a follows b, so content flows b -> a's feed.
    Returns (N, indptr, in_idx, follower_count, outdeg) where:
      audience[v] = in_idx[indptr[v]:indptr[v+1]] = v's FOLLOWERS (who see v's posts),
      follower_count[v] = in-degree (audience size; used to pick seeds),
      outdeg[u] = # accounts u FOLLOWS (u's feed breadth; sets its feed size / dilution)."""
    df = pd.read_csv(path, sep=" ", header=None, names=["src", "dst"], compression="gzip")
    nodes = pd.Index(pd.unique(pd.concat([df.src, df.dst], ignore_index=True)))
    code = {n: i for i, n in enumerate(nodes)}
    src = df.src.map(code).to_numpy(); dst = df.dst.map(code).to_numpy()
    N = len(nodes)
    order = np.argsort(dst, kind="stable")
    dst_s, src_s = dst[order], src[order]
    indptr = np.searchsorted(dst_s, np.arange(N + 1))
    follower_count = np.diff(indptr)                      # in-degree = # followers (audience)
    outdeg = np.bincount(src, minlength=N)                # out-degree = # followees (feed breadth)
    return N, indptr, src_s.astype(np.int64), follower_count, outdeg


def simulate_edge_feed(seed, indptr, in_idx, outdeg, N, pcurves, sar, install_rate, rng,
                       sar_k=SAR_DECAY_K_FALLBACK, post_rate=POST_RATE,
                       mix_sars=None, mix_w=None, mix_installs=None, mix_sar_ks=None):
    """One edge-mediated cascade with a random-order neighbour feed. Returns (reach, inst):
    reach = distinct agents who SAW the payload; inst = installs.

    Retransmission is RANK-RESOLVED: an agent that sees the payload at feed rank r retransmits so
    that the MARGINAL rate = sar0 * exp(-sar_k * r) (the measured SAR-by-rank decay, folded in via
    min(1, sar0*exp(-sar_k*r)/see) among those who saw). sar_k from sar_by_rank_<tag> (fallback =
    gpt-oss's measured decay). Buried payloads (high rank) are both seen less AND retransmitted less."""
    win_hb = WINDOW_H / CYCLE_H
    exposed = np.zeros(N, bool); exposed[seed] = True
    frontier = np.array([seed]); reach = 0; inst = 0
    for _ in range(200):                                   # generation cap (safety)
        segs = [in_idx[indptr[u]:indptr[u + 1]] for u in frontier]
        foll = np.concatenate(segs) if segs else np.empty(0, np.int64)
        if foll.size == 0:
            break
        cand = np.unique(foll); cand = cand[~exposed[cand]]
        if cand.size == 0:
            break
        exposed[cand] = True
        # random-order neighbour feed: payload at a uniform-random rank among the feed's posts
        F = 1 + rng.poisson(outdeg[cand] * post_rate * win_hb)
        r = (rng.random(cand.size) * F).astype(int)        # uniform rank in [0, F)
        rc = np.minimum(r, K_FEED - 1)
        see = np.where(r < K_FEED, curve(pcurves, "spread_any", rc), 0.0)
        saw = rng.random(cand.size) < see
        reached = cand[saw]
        if reached.size == 0:
            break
        reach += reached.size
        see_s = see[saw]; r_s = rc[saw]                    # rank + see of those who saw
        if mix_sars is not None:                           # heterogeneous population
            k = rng.integers(0, len(mix_sars), reached.size) if mix_w is None else \
                np.searchsorted(np.cumsum(mix_w), rng.random(reached.size))
            ir = np.asarray(mix_installs)[k]
            sar0 = np.asarray(mix_sars)[k]; sk = np.asarray(mix_sar_ks)[k]
            inst += int((rng.random(reached.size) < ir).sum())
            p_rt = np.minimum(1.0, sar0 * np.exp(-sk * r_s) / np.maximum(see_s, 1e-9))
        else:
            inst += int((rng.random(reached.size) < install_rate).sum())
            p_rt = np.minimum(1.0, sar * np.exp(-sar_k * r_s) / np.maximum(see_s, 1e-9))
        ret = reached[rng.random(reached.size) < p_rt]
        frontier = ret
    return reach, inst


def _tx(tag, arm):
    """P(payload|post) for an arm: Trojan = cross_child_<tag>_sec_mg, market-tip =
    cross_child_<tag>_alpha_af (weak control carrier), generic = cross_bare_<tag>_gen."""
    if arm == "Trojan":
        fname = f"cross_child_{tag}_sec_mg.json"
    elif arm == "market-tip":
        fname = f"cross_child_{tag}_alpha_af.json"
    else:
        fname = f"cross_bare_{tag}_gen.json"
    p = os.path.join("out/exposure", fname)
    d = json.load(open(p)); d = d[0] if isinstance(d, list) else d
    return float(d["p_payload_given_post"])


def _sar_decay_k(tag, fallback=SAR_DECAY_K_FALLBACK):
    """Rank-decay rate of retransmission SAR from sar_by_rank_<tag>.json (log-linear fit to
    sar_payload_permille by feed position). Returns (k, measured?). Fallback = gpt-oss's measured
    decay for models whose SAR-by-rank was not collected."""
    p = os.path.join("out/exposure", f"sar_by_rank_{tag}.json")
    if not os.path.exists(p):
        return fallback, False
    d = json.load(open(p))
    pos = np.array([r["position"] for r in d], float)
    sar = np.array([max((r.get("sar_payload_permille") or 0.0), 0.01) for r in d])
    if len(pos) < 2:
        return fallback, False
    _, slope = np.linalg.lstsq(np.vstack([np.ones_like(pos), pos]).T, np.log(sar), rcond=None)[0]
    return max(0.0, float(-slope)), True


def _pick(indeg, seed_class, rng, tol=0.15):
    """Seed picker: p10/p50/p90 = a random node whose follower count sits near that percentile of
    the connected (indeg>0) nodes — seed connectivity is a big lever, so we bracket it. Falls back
    to attack_reach_sim.pick_seed for leaf/median/hub."""
    if isinstance(seed_class, str) and seed_class.startswith("p") and seed_class[1:].isdigit():
        pos = indeg[indeg > 0]
        target = np.percentile(pos, int(seed_class[1:]))
        cand = np.where((indeg >= target * (1 - tol)) & (indeg <= target * (1 + tol)))[0]
        if cand.size == 0:
            cand = np.array([int(np.argmin(np.abs(indeg - target)))])
        return int(rng.choice(cand))
    return pick_seed(indeg, seed_class, rng)


def _dist(reps, seed_class, sar, install_rate, indptr, in_idx, outdeg, N, indeg, pcurves, rng,
          sar_k=SAR_DECAY_K_FALLBACK, mix_sars=None, mix_w=None, mix_installs=None, mix_sar_ks=None):
    reach = np.empty(reps); inst = np.empty(reps)
    for i in range(reps):
        s = _pick(indeg, seed_class, rng)
        cr, ci = simulate_edge_feed(s, indptr, in_idx, outdeg, N, pcurves, sar, install_rate, rng,
                                    sar_k=sar_k, mix_sars=mix_sars, mix_w=mix_w,
                                    mix_installs=mix_installs, mix_sar_ks=mix_sar_ks)
        reach[i] = cr; inst[i] = ci
    return reach, inst


def run(reps=1000, seed_class="median", seed0=0):
    os.makedirs(OUT, exist_ok=True); os.makedirs(FIGDIR, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    rng = np.random.default_rng(seed0)
    print("loading ego-Twitter follower graph...", flush=True)
    N, indptr, in_idx, indeg, outdeg = load_follower_graph()
    print(f"N={N:,} | edges={in_idx.size:,} | mean followers={indeg.mean():.1f} mean followees={outdeg.mean():.1f} | "
          f"reps={reps} seed_class={seed_class}", flush=True)

    # per-model curves, install rate, arm SARs, retransmission rank-decay
    pcs, insts, sarT, sarG, sark = {}, {}, {}, {}, {}
    for name, tag in MODELS:
        pcs[name] = _pcurves_with_payload(unified_for(tag))
        ir = install_rank0(tag); insts[name] = ir if ir is not None else 0.66
        sarT[name] = P_POST_REF * _tx(tag, "Trojan")
        sarG[name] = P_POST_REF * _tx(tag, "Generic")
        sark[name], meas = _sar_decay_k(tag)
        print(f"  {name}: install(rank0)={insts[name]:.2f}{'(fallback)' if ir is None else ''}  "
              f"SAR Trojan={sarT[name]*1000:.1f}permille generic={sarG[name]*1000:.1f}permille  "
              f"SAR-decay k={sark[name]:.3f}{'(measured)' if meas else '(fallback=gpt-oss)'}", flush=True)
    mean_install = float(np.mean([insts[n] for n, _ in MODELS]))
    mean_k = float(np.mean([sark[n] for n, _ in MODELS]))

    raw = {}; rows = []
    # non-transmitted baseline (sar=0): only the seed's directly-reached followers, no cascade
    rb, ib = _dist(reps, seed_class, 0.0, mean_install, indptr, in_idx, outdeg, N, indeg,
                   pcs[MODELS[0][0]], rng)
    raw["non-transmitted baseline"] = (rb, ib); rows.append(_row("non-transmitted baseline", 0.0, rb, ib))
    print(f"  baseline: reach_mean={rb.mean():.1f} inst_mean={ib.mean():.1f}", flush=True)

    for name, tag in MODELS:
        for arm, sar in [("Trojan", sarT[name]), ("Generic", sarG[name])]:
            r, i = _dist(reps, seed_class, sar, insts[name], indptr, in_idx, outdeg, N, indeg, pcs[name],
                         rng, sar_k=sark[name])
            lbl = f"{name}/{arm}"; raw[lbl] = (r, i); rows.append(_row(lbl, sar * 1000, r, i))
            print(f"  {lbl:28s} SAR={sar*1000:6.1f}permille  reach_mean={r.mean():8.1f}  inst_mean={i.mean():8.1f}",
                  flush=True)

    # equal-share mixture (Trojan arm) — heterogeneous population over all models
    nm = len(MODELS)
    mix_sars = [sarT[n] for n, _ in MODELS]; mix_w = [1.0 / nm] * nm
    mix_installs = [insts[n] for n, _ in MODELS]; mix_sar_ks = [sark[n] for n, _ in MODELS]
    rm, im = _dist(reps, seed_class, 0.0, mean_install, indptr, in_idx, outdeg, N, indeg, pcs[MODELS[0][0]],
                   rng, mix_sars=mix_sars, mix_w=mix_w, mix_installs=mix_installs, mix_sar_ks=mix_sar_ks)
    raw["mixture (Trojan, equal)"] = (rm, im); rows.append(_row("mixture (Trojan, equal)", np.mean(mix_sars) * 1000, rm, im))
    print(f"  mixture (Trojan): reach_mean={rm.mean():.1f} inst_mean={im.mean():.1f}", flush=True)

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "reach_by_model_edge.csv"), index=False)
    rawrows = [{"model": m, "rep": k, "reach": rr, "inst": ii}
               for m, (ra, ia) in raw.items() for k, (rr, ii) in enumerate(zip(ra, ia))]
    pd.DataFrame(rawrows).to_csv(os.path.join(OUT, "reach_raw_edge.csv"), index=False)
    _amplification(raw)
    _plot_ccdf(raw, N)
    print(f"\nwrote {OUT}/reach_raw_edge.csv, reach_by_model_edge.csv, amplification_edge.csv, "
          f"{FIGDIR}/reach_edge_mediated.pdf, {FIG}/tab_amplification_edge.tex")


def _row(name, sar_permille, reach, inst):
    row = {"model": name, "SAR_permille": round(float(sar_permille), 2),
           "reach_mean": round(float(reach.mean()), 1), "reach_p99": round(float(np.percentile(reach, 99)), 1),
           "inst_mean": round(float(inst.mean()), 1), "inst_p99": round(float(np.percentile(inst, 99)), 1)}
    for x in (10, 100, 1000):
        row[f"P_reach>{x}"] = round(float((reach > x).mean()), 3)
    for x in (1, 10, 100):
        row[f"P_inst>{x}"] = round(float((inst > x).mean()), 3)
    return row


def _amplification(raw):
    rows = []
    for name, _ in MODELS:
        T, G = raw[f"{name}/Trojan"], raw[f"{name}/Generic"]
        aR = T[0].mean() / max(G[0].mean(), 1e-9); aI = T[1].mean() / max(G[1].mean(), 1e-9)
        rows.append(dict(model=name, trojan_reach=round(T[0].mean(), 1), generic_reach=round(G[0].mean(), 1),
                         amp_reach=round(aR, 2), trojan_inst=round(T[1].mean(), 1),
                         generic_inst=round(G[1].mean(), 1), amp_inst=round(aI, 2)))
        print(f"  AMPLIFICATION {name}: reach x{aR:.2f}  installs x{aI:.2f}", flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "amplification_edge.csv"), index=False)
    lines = [
        "% Edge-mediated (follower-graph) attack amplification: Trojan vs bare generic link.",
        "% Random-order neighbour feed (no karma) -> pure retransmission. sandbox.reach_edge.",
        "\\begin{table}[t]\\centering\\small",
        "\\caption{\\textbf{Edge-mediated attack amplification} (SNAP ego-Twitter follower graph, "
        "$N{=}81{,}306$): memetic Trojan vs a bare generic link carrying the same payload. The feed is "
        "each follower's neighbours' posts in \\emph{random} order, so there is no karma channel and "
        "amplification is driven purely by retransmission (SAR) $\\times$ graph structure. Amplification "
        "$=$ Trojan$/$generic.}",
        "\\label{tab:amp-edge}", "\\begin{tabular}{l rr r rr r}", "\\toprule",
        "Model & \\multicolumn{2}{c}{reach} & amp & \\multicolumn{2}{c}{installs} & amp\\\\",
        " & Trojan & generic & $\\times$ & Trojan & generic & $\\times$\\\\", "\\midrule"]
    for r in rows:
        lines.append(f"{r['model'].replace('_','\\_')} & {r['trojan_reach']:.0f} & {r['generic_reach']:.0f} "
                     f"& {r['amp_reach']:.2f} & {r['trojan_inst']:.0f} & {r['generic_inst']:.0f} & {r['amp_inst']:.2f}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    open(os.path.join(FIG, "tab_amplification_edge.tex"), "w").write("\n".join(lines))


def _plot_ccdf(raw, N):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base = raw["non-transmitted baseline"][1]
    b_mean, b_p99 = base.mean(), np.percentile(base, 99)
    allmax = max(v[1].max() for v in raw.values())
    xmax = max(float(allmax), 100.0, N)
    xs = np.unique(np.round(np.logspace(0, np.log10(xmax) + 0.05, 90)).astype(int))
    n = len(base); floor = 0.5 / max(n, 1)
    def ccdf(a):
        y = np.array([(a > x).mean() for x in xs]); return np.where(y > 0, y, np.nan)
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for name, _ in MODELS:                                  # Trojan arms (generic is in the table)
        ax.plot(xs, ccdf(raw[f"{name}/Trojan"][1]), color=COL[name], lw=2, label=f"{name} (Trojan)")
    ax.plot(xs, ccdf(raw["mixture (Trojan, equal)"][1]), color="black", lw=2.2, ls=(0, (4, 2)),
            label="mixture (Trojan)")
    ax.axvline(b_mean, color="0.4", ls="--", lw=1.1)
    ax.text(b_mean, 0.5, f"baseline mean ({b_mean:.0f})", fontsize=7, color="0.3", rotation=90,
            ha="right", va="center", transform=ax.get_xaxis_transform())
    ax.axvline(N, color="#2a7", ls="-.", lw=1.4)
    ax.text(N, 0.5, f"$N={N:,}$", fontsize=7, color="#178", rotation=90, ha="right", va="center",
            transform=ax.get_xaxis_transform())
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1, xmax * 1.15); ax.set_ylim(floor, 1.3)
    ax.set_xlabel("x  (payload installs)"); ax.set_ylabel("P(payload installs > x)")
    ax.set_title("Edge-mediated cascade (ego-Twitter follower graph): payload installs\n"
                 "random-order neighbour feed — pure retransmission, Trojan vs generic", fontsize=10)
    ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "reach_edge_mediated.pdf"), bbox_inches="tight")
    plt.close(fig)


def _boot_ci(x, nboot=4000, seed=0):
    """mean and 95% CI of the mean via bootstrap over per-run values."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    bs = np.array([rng.choice(x, x.size).mean() for _ in range(nboot)])
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def _boot_ratio_ci(a, b, nboot=4000, seed=0):
    """mean and 95% CI of mean(a)/mean(b) via paired-independent bootstrap (a,b are separate arrays)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float); b = np.asarray(b, float)
    rs = np.array([rng.choice(a, a.size).mean() / max(rng.choice(b, b.size).mean(), 1e-9) for _ in range(nboot)])
    return float(a.mean() / max(b.mean(), 1e-9)), float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def _pct(x):
    """(median, p10, p90) of the OUTCOME distribution — cascade sizes are heavy-tailed, so the spread
    of realizations (not the CI of the mean) is what conveys the tail."""
    x = np.asarray(x, float)
    return float(np.median(x)), float(np.percentile(x, 10)), float(np.percentile(x, 90))


def run_seed_protos(reps=3000, seed0=0, protos=("p99", "p90", "p50", "p10")):
    """Edge cascade by SEED CONNECTIVITY prototype (p99/p90/p50/p10 follower-count percentile), all
    models, Trojan vs generic. Reach/installs are reported as the OUTCOME distribution across `reps`
    cascades (median [p10-p90]) — follower cascades are heavy-tailed, so most fizzle and a few explode;
    the median+p10/p90 shows that tail, where a CI on the mean would hide it. Also reports the
    NON-TRANSMITTED baseline installs (sar=0: the seed's direct followers only, no re-transmission) and
    the Trojan/generic amplification (ratio of means, bootstrap 95% CI)."""
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(seed0)
    N, indptr, in_idx, indeg, outdeg = load_follower_graph()
    pos = indeg[indeg > 0]
    pctl = {p: float(np.percentile(pos, int(p[1:]))) for p in protos}
    print(f"N={N:,} | reps={reps} | follower-count percentiles: "
          + ", ".join(f"{p}={v:.0f}" for p, v in pctl.items()), flush=True)
    rows = []
    for name, tag in MODELS:
        pc = _pcurves_with_payload(unified_for(tag))
        ir = install_rank0(tag); inst = ir if ir is not None else 0.66
        sarT = P_POST_REF * _tx(tag, "Trojan"); sarG = P_POST_REF * _tx(tag, "Generic")
        k, _meas = _sar_decay_k(tag)
        for proto in protos:
            rT, iT = _dist(reps, proto, sarT, inst, indptr, in_idx, outdeg, N, indeg, pc, rng, sar_k=k)
            rG, iG = _dist(reps, proto, sarG, inst, indptr, in_idx, outdeg, N, indeg, pc, rng, sar_k=k)
            rB, iB = _dist(reps, proto, 0.0, inst, indptr, in_idx, outdeg, N, indeg, pc, rng, sar_k=k)  # non-transmitted
            rmed, rp10, rp90 = _pct(rT); imed, ip10, ip90 = _pct(iT); bmed, bp10, bp90 = _pct(iB)
            am, alo, ahi = _boot_ratio_ci(rT, rG); aim, ailo, aihi = _boot_ratio_ci(iT, iG)
            rows.append(dict(model=name, seed=proto, followers=round(pctl[proto]), reps=reps,
                             reach=round(rmed, 1), reach_lo=round(rp10, 1), reach_hi=round(rp90, 1),
                             installs=round(imed, 1), installs_lo=round(ip10, 1), installs_hi=round(ip90, 1),
                             base_installs=round(bmed, 1), base_installs_lo=round(bp10, 1), base_installs_hi=round(bp90, 1),
                             amp_reach=round(am, 2), amp_reach_lo=round(alo, 2), amp_reach_hi=round(ahi, 2),
                             amp_inst=round(aim, 2), amp_inst_lo=round(ailo, 2), amp_inst_hi=round(aihi, 2),
                             generic_reach=round(float(rG.mean()), 1), install_fallback=(ir is None)))
            print(f"  {name:16s} {proto} (foll~{pctl[proto]:.0f})  reach med={rmed:7.1f}[{rp10:.0f},{rp90:.0f}]  "
                  f"inst med={imed:7.1f}[{ip10:.0f},{ip90:.0f}]  base(no-retx)={bmed:.1f}  amp={am:.2f}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "edge_seed_protos.csv"), index=False)
    _plot_seed_protos(rows)
    print(f"\nwrote {OUT}/edge_seed_protos.csv, {FIGDIR}/edge_seed_protos.pdf")


def _plot_seed_protos(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.DataFrame(rows)
    protos = ["p10", "p50", "p90", "p99"]                       # ascending connectivity
    xpos = {p: i for i, p in enumerate(protos)}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for name in df.model.unique():
        s = df[df.model == name].set_index("seed")
        x = [xpos[p] for p in protos if p in s.index]
        # reach with CI (log)
        rm = [s.loc[p, "reach"] for p in protos if p in s.index]
        rlo = [s.loc[p, "reach"] - s.loc[p, "reach_lo"] for p in protos if p in s.index]
        rhi = [s.loc[p, "reach_hi"] - s.loc[p, "reach"] for p in protos if p in s.index]
        ax1.errorbar(x, rm, yerr=[rlo, rhi], marker="o", lw=1.8, ms=4, capsize=3, label=name)
        # amp with CI
        am = [s.loc[p, "amp_reach"] for p in protos if p in s.index]
        alo = [s.loc[p, "amp_reach"] - s.loc[p, "amp_reach_lo"] for p in protos if p in s.index]
        ahi = [s.loc[p, "amp_reach_hi"] - s.loc[p, "amp_reach"] for p in protos if p in s.index]
        ax2.errorbar(x, am, yerr=[alo, ahi], marker="s", lw=1.8, ms=4, capsize=3, label=name)
    for ax, ttl, yl in [(ax1, "Absolute Trojan reach (95% CI)", "reach (agents)"),
                        (ax2, "Amplification Trojan/generic (95% CI)", "reach amp ×")]:
        ax.set_xticks(range(len(protos))); ax.set_xticklabels(protos)
        ax.set_xlabel("seed connectivity (follower-count percentile)"); ax.set_ylabel(yl)
        ax.set_title(ttl, fontsize=10); ax.grid(alpha=0.25, which="both"); ax.spines[["top", "right"]].set_visible(False)
    ax1.set_yscale("log"); ax2.axhline(1.0, color="0.6", ls=":", lw=1)
    ax1.legend(fontsize=7.5, title="model"); fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "edge_seed_protos.pdf"), bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed-class", default="median", choices=["leaf", "median", "hub"])
    ap.add_argument("--seed-protos", action="store_true", help="run p90/p50/p10 seed-connectivity prototypes")
    ap.add_argument("--seed0", type=int, default=0)
    args = ap.parse_args()
    if args.seed_protos:
        run_seed_protos(reps=args.reps, seed0=args.seed0)
    else:
        run(reps=args.reps, seed_class=args.seed_class, seed0=args.seed0)
