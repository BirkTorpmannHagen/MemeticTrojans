"""Regime B — STATE-MEDIATED (endogenous feed-loop) contagion phase diagram.

MODEL VERSION: docs/SIMULATION_MODEL.md v2. Provides both simulate_endo (phase diagram)
and simulate_surface_cascade (the surface x cascade engine used by expected_installs_surface).

The headline mechanism result: with Moltbook's KNOWN ranking operator H
(recency + 0.2*upvotes) fixed, a small measured per-encounter retransmission
probability can be sub- or super-critical depending on how strongly agents
AMPLIFY (upvote). This is the endogenous counterpart to the edge-mediated cascade
sims (attack_reach_sim.py = Regime A): same agent susceptibility, different
amplification operator (ranking vs topology). See README.

Loop (rank -> expose -> respond -> vote/retransmit -> re-rank), per 30-min cycle:
  * All live posts are ranked by H; the top-K is the global heartbeat feed shown
    to the whole susceptible population (broadcast — the feed is a shared state).
  * A meme post at feed rank r exposes a fraction see(r) of remaining susceptibles
    (exposure geometry fixed from the assay's rank-decay of spread_any).
  * Each exposed agent upvotes with prob p_vote (amplification; raises the post's
    score -> keeps it in the feed longer -> the endogenous loop) and retransmits a
    fresh meme post with prob p_ret (the measured per-encounter SAR ~0.01 at rank 0).
  * Background competitors co-evolve and turn over, competing for the K feed slots.

We SWEEP (p_ret, p_vote) with H, K and the exposure geometry held fixed, and map
the extinction -> macroscopic-prevalence phase boundary. The measured operating
point (p_ret=payload_post@rank0, p_vote=upvote@rank0 from the unified assay) is
placed on the diagram.

    python -m sandbox.endogenous_sim --smoke     # 4x4 grid, fast
    python -m sandbox.endogenous_sim             # full grid + figure

Outputs: out/endogenous_sim/phase_grid.csv, figures/fig_endogenous_phase.pdf.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from analysis import load
from sandbox.attack_reach_sim import (
    K_FEED, HALF_LIFE_H, CYCLE_H, WINDOW_H, load_params, curve,
)

OUT = "out/endogenous_sim"
FIGDIR = "figures"
UNIFIED = "out/exposure/unified_gptoss120b.json"


# Reliable per-model payload-transmission SAR (prefill decomposition; security_warning,
# the dominant carrier). Source: out/exposure/sampled_sar_qwen*.json + ollama_sar_*.json
# SAR is p_ret; dominated by activity P(post).
SAR_MODELS = {
    "gpt-oss-120B": 0.0125, "qwen-32B": 0.0225, "qwen-3B": 0.0347,
    "qwen-14B": 0.0375, "deepseek-v4-flash": 0.178,
}


def fitted_half_life(params="out/exposure/exposure_params.json", fallback=HALF_LIFE_H):
    """Engagement/recency half-life measured from live snapshots (Workstream 2c),
    replacing the nominal 3h. Falls back to the nominal value if not yet estimated.
    NOTE: censoring-biased UP and measured at ~4.6x lower activity than the corpus."""
    try:
        return float(json.load(open(params))["engagement_half_life_h"])
    except Exception:
        return fallback


def measured_point(unified=UNIFIED):
    """(p_ret, p_vote) at feed rank 0 from the unified single-agent assay."""
    u = pd.DataFrame(json.load(open(unified)))
    g = u.groupby("position")[["payload_post", "upvote"]].mean()
    return float(g.loc[0, "payload_post"]), float(g.loc[0, "upvote"])


@dataclass(frozen=True)
class SimConfig:
    """Single shared configuration for the state-mediated feed-cascade reach model.
    Every reach/install generator (reach_by_model, expected_installs_surface) imports
    CONFIG so they run the SAME model. See docs/SIMULATION_MODEL.md for the full
    assumption ledger.

    Exposure is counted in VIEWS: a post at feed rank r is seen by
    ``spread_any(r) * N`` agents that heartbeat (state-mediated broadcast — a
    top-feed post reaches ~all of N). A freshly seeded, 0-upvote payload post is
    buried under the corpus-karma competitor field and only reaches the top slots
    (where views explode) if amplification + retransmission carry it there
    (EARNED CLIMB). Its guaranteed foothold is the bottom visible feed slot
    ``coldstart_rank`` (the "new/rising" surface)."""
    N: int = 37189                # total population (Moltbook agents; fixed)
    cycles: int = 120             # 30-min heartbeat cycles simulated
    n_live: int = 6685            # competitor live-post field (corpus posts/cycle * 6h window)
    coldstart_rank: int = K_FEED - 1   # a fresh post's guaranteed slot = bottom of feed (legacy)
    bcast_frac: float = 0.94      # top-slot BROADCAST exposure: fraction of heartbeating agents who
                                  #   see a post holding the top of a ranked surface. Approx the
                                  #   measured spread_any(0) across current models (0.93-0.99;
                                  #   unified_{gptoss120b,deepseekflash}). NOT gpt-4o-mini.
    coldstart_frac: float = 0.033  # legacy foothold (used by simulate_endo phase diagram only).
    new_read_frac: float = 0.05   # phi_new: fraction of agents who browse the chronological `new`
                                  #   feed (the default feed is `hot`). A BURIED post can only reach
                                  #   this subpopulation, so it CAPS buried reach at new_read_frac*N.
                                  #   SENSITIVITY knob (sweep 0.01-0.25). Not snapshot-measurable
                                  #   (agent behaviour); the structural factors below are measured.
    ambient_new: float = 38.0     # background fresh posts entering the `new` top-25 per 30-min cycle
                                  #   (measured: 25 per 20-min poll = 75/h, moltbook_snapshots). A
                                  #   cascade with C live fresh posts occupies share C/(C+ambient_new)
                                  #   of the new feed, so buried exposure scales with cascade size.
    view_upvote: float = 0.02     # realized fraction of VIEWERS who upvote (amplification).
                                  #   NOT the assay's forced P(upvote|seen)=0.73; at any
                                  #   plausible value (0-0.1) reach is insensitive — the
                                  #   endogenous upvote loop is negligible vs cold-start x SAR.
    maxchildren: int = 400        # per-cycle retransmission cap (numerical stability)


CONFIG = SimConfig()


def _view_shape(seecurves):
    """Per-rank view weight = spread_any(r)/spread_any(0), normalized to 1.0 at the top slot.
    Retained only for the legacy upvoter-scale `endogenous_sim_conserved` variant (used by
    sandbox.make_install_tables); the unified views-based engine below does not use it."""
    w = np.array([float(curve(seecurves, "spread_any", r)) for r in range(K_FEED)])
    return w / w[0]


def simulate_endo(p_ret, p_vote, seecurves, corpus_up, rng, N=None, cycles=None,
                  half_life=HALF_LIFE_H, mix_sars=None, mix_w=None,
                  install_rate=None, mix_installs=None, cfg=CONFIG, pin_rank=None):
    """One state-mediated feed-cascade run (VIEWS-based, earned-climb).

    pin_rank    RANK-CONDITIONAL mode (assumption-light alternative to the earned climb):
                if given, EVERY live payload post is treated as sitting at feed rank
                `pin_rank`, so views = spread_any(pin_rank) * remaining_sus each cycle,
                with no competitor ranking, cold-start, or upvote climb. Sweeping pin_rank
                answers "if a seed post holds rank r, how far does the payload spread?"
                without modelling how it gets there.

    p_ret       per-encounter retransmission probability = SAR = P(post)*P(payload|post),
                measured per model (real data). A viewer retransmits (authors a fresh
                payload post) with this probability.
    p_vote      kept for signature/back-compat with the phase-diagram sweep; the realized
                view->upvote amplification rate actually used is cfg.view_upvote (the assay
                p_vote/0.73 is a forced-attention rate, not a per-viewer platform rate).
    seecurves   assay rank curves (spread_any/install/upvote) from load_params.
    install_rate  P(install | seen) at rank 0 (per-model ASR); required (no legacy fallback).
    mix_sars/mix_w/mix_installs  heterogeneous population: each cold-start viewer is a
                random model from the mix and retransmits / installs at that model's rate.

    Mechanism per 30-min cycle, for each live payload post at feed rank r:
      * VIEWS = spread_any(r) * (remaining susceptibles); a fresh post (age 0) is floored
        at the bottom-slot visibility spread_any(coldstart_rank) = its new-feed foothold.
      * those viewers INSTALL w.p. install_rate  -> cum_inst
      * those viewers RETRANSMIT w.p. p_ret       -> fresh payload posts (branching)
      * those viewers UPVOTE w.p. cfg.view_upvote -> post score climbs (usually too weak
        to crack the corpus-karma feed, so the loop is second-order).
    Reach depletes a shared susceptible pool of size N (no re-exposure counted).

    Returns (cum_reach distinct viewers, cum_inst installs, peak_live meme posts, persisted?).
    """
    N = cfg.N if N is None else int(N)
    cycles = cfg.cycles if cycles is None else int(cycles)
    if install_rate is None:
        raise ValueError("install_rate is required (P(install|seen) at rank 0)")
    N = int(N)
    maxage = int(WINDOW_H / CYCLE_H) + 2
    f_new = float(curve(seecurves, "spread_any", cfg.coldstart_rank))   # bottom-slot foothold
    pinned_vis = None if pin_rank is None else float(curve(seecurves, "spread_any", pin_rank))

    # Static competitor field: corpus-karma live posts, used only to RANK meme posts.
    # score = recency(age in window) + 0.2*upvotes  (Moltbook ranking operator H).
    comp = None if pin_rank is not None else np.sort(
        1.0 / (1.0 + rng.uniform(0, WINDOW_H, cfg.n_live) / half_life)
        + 0.2 * rng.choice(corpus_up, cfg.n_live))

    age = np.array([0.0]); up = np.array([0.0])     # one seed payload post, buried, 0 upvotes
    remaining_sus = float(N)
    cum_reach = 0.0; cum_inst = 0.0
    peak_live = 1; persisted = False

    for t in range(cycles):
        if age.size == 0:
            break
        if pin_rank is not None:                     # rank-conditional: every post held at pin_rank
            vis = np.full(age.size, pinned_vis)
            in_feed = np.ones(age.size, bool)
        else:
            score = 1.0 / (1.0 + age * CYCLE_H / half_life) + 0.2 * up
            above = comp.size - np.searchsorted(comp, score, side="right")   # # posts outranking
            in_feed = above < K_FEED
            vis = np.where(in_feed, curve(seecurves, "spread_any", np.minimum(above, K_FEED - 1)), 0.0)
            vis = np.maximum(vis, np.where(age == 0, f_new, 0.0))            # cold-start foothold
        peak_live = max(peak_live, int(age.size))

        # distinct new reach this cycle. The feed is a SHARED global surface: an agent sees the
        # meme at its BEST-placed live post, NOT once per buried duplicate. So reach is driven by
        # the best feed position, max(vis) -- NOT 1-prod(1-vis), which would treat hundreds of
        # buried retransmitted copies as independent exposure chances and compound to fake
        # saturation. Attribute the new reach back to posts proportional to vis.
        p_any = float(vis.max()) if vis.size else 0.0
        new = remaining_sus * p_any
        if new <= 0:
            attrib = np.zeros_like(vis)
        else:
            wsum = vis.sum()
            attrib = (vis / wsum) * new if wsum > 0 else np.zeros_like(vis)
            remaining_sus -= new; cum_reach += new
        n_new = int(round(new))

        children = 0
        if n_new > 0:
            if mix_sars is not None:                    # heterogeneous population
                counts = rng.multinomial(n_new, mix_w)
                for k, sk in enumerate(mix_sars):
                    if counts[k]:
                        children += int(rng.binomial(counts[k], min(1.0, sk)))
                        ik = mix_installs[k] if mix_installs is not None else install_rate
                        cum_inst += counts[k] * ik
            else:
                children = int(rng.binomial(n_new, min(1.0, p_ret))) if p_ret > 0 else 0
                cum_inst += new * install_rate

        up = up + attrib * cfg.view_upvote              # amplification (realized upvote rate)
        age = age + 1.0
        alive = age <= maxage
        age, up = age[alive], up[alive]
        nc = min(children, cfg.maxchildren)             # fresh payload posts (branching)
        if nc:
            age = np.concatenate([age, np.zeros(nc)])
            up = np.concatenate([up, np.zeros(nc)])
        if remaining_sus <= 0:
            persisted = True
            break
        if t == cycles - 1:
            persisted = bool(in_feed.any())

    return cum_reach, cum_inst, peak_live, persisted


def simulate_surface_cascade(p_ret, install_rate, seed_bcast, seecurves, rng,
                             N=None, cycles=None, cfg=CONFIG, ret_decay=1.0):
    """State-mediated cascade with an explicit BROADCAST-SURFACE break-in (unified surface x
    cascade model; see docs/SIMULATION_MODEL.md and the expected-installs tables).

    Break-in is the ATTACKER'S SEED post catching fire onto a broadcast surface -- a single
    Bernoulli per seeded attack, passed in as `seed_bcast` (the caller sets it from the
    entry-bar probability p_break = P(peak*r_up >= tau_s)*timing_s for surface s and carrier
    r_up). Retransmitted children INHERIT the seed's broadcast status; giving each child its
    OWN break-in shot over the horizon saturates everything to N and destroys the tail.

    * seed_bcast=True  : the seed (and its inheriting children) hold the top feed slot
      spread_any(0) ~ 0.94 -> the cascade saturates to ~N*install (the viral outcome).
    * seed_bcast=False : buried. The post never reaches `hot`; it can only reach agents who browse
      the chronological `new` feed, a fraction cfg.new_read_frac of N -- so buried reach is CAPPED at
      new_read_frac*N. Each cycle the cascade's C live fresh posts occupy a share
      C/(C+cfg.ambient_new) of the `new` feed, and expose that share of the remaining new-readers.
      Exposure therefore scales with cascade size (a lone/dying cascade reaches few; only a cascade
      that floods `new` approaches the new_read_frac*N ceiling) -- not a fixed fraction each cycle.

    Surfaces and carriers enter the table ONLY through the caller's mix weight P(seed_bcast);
    the two outcome distributions (buried, broadcast) are surface/carrier-independent per model,
    so a whole table is two Monte-Carlo pools plus a per-cell Bernoulli mixture.

    Returns (cum_reach, cum_inst)."""
    N = cfg.N if N is None else int(N)
    cycles = cfg.cycles if cycles is None else int(cycles)
    maxage = int(WINDOW_H / CYCLE_H) + 2
    f_bcast = cfg.bcast_frac        # top-slot broadcast exposure (explicit config)
    # Broadcast reaches the whole population; buried reaches only the `new`-feed readership.
    pool = float(N) if seed_bcast else cfg.new_read_frac * float(N)

    age = np.array([0.0])
    remaining = pool; cum_reach = 0.0; cum_inst = 0.0
    for cyc in range(cycles):
        if age.size == 0:
            break
        # ret_decay<1 models per-generation attenuation of retransmission: each successive cascade
        # cycle is one more rephrasing removed from the pristine seed, so p_ret decays geometrically.
        pr = p_ret * (ret_decay ** cyc)
        # Broadcast posts hold the top slot (fixed exposure). Buried posts compete in `new`: the
        # cascade's C live fresh (age-0) posts occupy share C/(C+ambient_new) of the new feed, so
        # exposure scales with cascade size and is drawn from the new-reader pool only.
        if seed_bcast:
            p_any = f_bcast
        else:
            c_live = float((age == 0).sum())
            p_any = c_live / (c_live + cfg.ambient_new) if c_live > 0 else 0.0
        new = remaining * p_any
        if new > 0:
            remaining -= new; cum_reach += new; cum_inst += new * install_rate
        n_new = int(round(new))
        children = int(rng.binomial(n_new, min(1.0, pr))) if (n_new > 0 and pr > 0) else 0
        age = (age + 1.0)[age + 1.0 <= maxage]
        nc = min(children, cfg.maxchildren)
        if nc:
            age = np.concatenate([age, np.zeros(nc)])
        if remaining <= 0:
            break
    return cum_reach, cum_inst


def run(smoke=False, seed=0, N=None, cycles=None, reps=6):
    N = CONFIG.N if N is None else N
    cycles = CONFIG.cycles if cycles is None else cycles
    os.makedirs(OUT, exist_ok=True); os.makedirs(FIGDIR, exist_ok=True)
    rng = np.random.default_rng(seed)
    seecurves = load_params(UNIFIED)                 # exposure geometry (spread_any etc.)
    corpus_up = load.load_posts()["upvotes"].dropna().to_numpy()
    corpus_up = corpus_up[corpus_up >= 0].astype(float)
    m_ret, m_vote = measured_point()
    half_life = fitted_half_life()                    # snapshot-measured (2c), fallback nominal 3h
    print(f"half_life={half_life:.1f}h (nominal {HALF_LIFE_H}h) | "
          f"reliable SAR range {min(SAR_MODELS.values())*1000:.0f}-{max(SAR_MODELS.values())*1000:.0f} permille",
          flush=True)

    # p_vote axis is now the REALIZED view->upvote amplification rate (cfg.view_upvote),
    # swept from realistic (few %) to unrealistic (95%) to locate where the endogenous
    # loop starts to matter. Reach is dominated by p_ret x cold-start until it is large.
    if smoke:
        p_ret_grid = np.array([0.002, 0.0225, 0.0375, 0.178])
        p_vote_grid = np.array([0.02, 0.1, 0.4, 0.9])
        reps = 3
    else:
        # grid spans the reliable per-model SAR band (12.5-178 permille)
        p_ret_grid = np.array([0.001, 0.003, 0.006, 0.0125, 0.0225, 0.0375, 0.06, 0.10, 0.178])
        p_vote_grid = np.array([0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.95])

    inst0 = float(curve(seecurves, "install", 0) / curve(seecurves, "spread_any", 0))
    rows = []
    for pr in p_ret_grid:
        for pv in p_vote_grid:
            cfg = SimConfig(N=N, cycles=cycles, view_upvote=float(pv))
            reach = []; persist = []
            for rep in range(reps):
                cr, _ci, pk, ps = simulate_endo(pr, pv, seecurves, corpus_up, rng,
                                                half_life=half_life, install_rate=inst0, cfg=cfg)
                reach.append(cr); persist.append(ps)
            rows.append({
                "p_ret": pr, "p_vote": pv,
                "prevalence": float(np.mean(reach)) / N,
                "persist_frac": float(np.mean(persist)),
                "reach_mean": float(np.mean(reach)),
            })
            print(f"  p_ret={pr:.4f} p_vote={pv:.2f} -> prevalence="
                  f"{np.mean(reach)/N:.3f} persist={np.mean(persist):.2f}", flush=True)

    df = pd.DataFrame(rows)
    path = os.path.join(OUT, "phase_grid.csv")
    df.to_csv(path, index=False)
    print(f"wrote {path}")
    _plot(df, p_ret_grid, p_vote_grid, m_ret, m_vote,
          os.path.join(FIGDIR, "fig_endogenous_phase.pdf"), half_life=half_life)


def _plot(df, p_ret_grid, p_vote_grid, m_ret, m_vote, path, half_life=HALF_LIFE_H):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm

    piv = (df.pivot(index="p_ret", columns="p_vote", values="prevalence")
             .reindex(index=p_ret_grid, columns=p_vote_grid))
    Z = piv.to_numpy()
    ny, nx = Z.shape

    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    # categorical cells (even spacing) with a gamma-stretched map so the boundary shows
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap="magma",
                   norm=PowerNorm(gamma=0.5, vmin=0, vmax=1),
                   extent=[-0.5, nx - 0.5, -0.5, ny - 0.5])
    ax.set_xticks(range(nx)); ax.set_yticks(range(ny))
    ax.set_xticklabels([f"{v:.2f}" for v in p_vote_grid])
    ax.set_yticklabels([f"{v:.3f}" for v in p_ret_grid])
    # annotate each cell with its prevalence
    for i in range(ny):
        for j in range(nx):
            v = Z[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if v < 0.5 else "black")

    # criticality boundary (prevalence = 0.05) in cell coordinates
    try:
        gx, gy = np.meshgrid(np.arange(nx), np.arange(ny))
        cs = ax.contour(gx, gy, Z, levels=[0.05], colors="cyan", linewidths=2.0)
        ax.clabel(cs, fmt={0.05: "criticality"}, fontsize=8)
    except Exception:
        pass

    # Reliable per-model SAR (prefill decomposition) as horizontal p_ret levels.
    # p_vote is not measured per sweep-model, so we draw across the p_vote axis.
    def y_of(pr):
        return float(np.interp(np.log(pr), np.log(p_ret_grid), np.arange(ny)))
    sar_vals = sorted(SAR_MODELS.values())
    ax.axhspan(y_of(sar_vals[0]), y_of(sar_vals[-1]), color="cyan", alpha=0.10, zorder=1)
    for name, pr in sorted(SAR_MODELS.items(), key=lambda kv: kv[1]):
        yy = y_of(pr)
        ax.axhline(yy, color="cyan", lw=1.0, ls=":", alpha=0.8, zorder=4)
        ax.text(nx - 0.35, yy, f"{name} ({pr*1000:.0f}‰)", fontsize=6.5,
                color="#0aa", va="center", ha="left", zorder=6)
    ax.set_xlim(-0.5, nx - 0.5 + 3.2)   # room for right-edge SAR labels

    ax.set_xlabel(r"$p_{vote}$  — amplification (upvote prob per exposure, raises feed rank)")
    ax.set_ylabel(r"$p_{ret}$  — per-encounter retransmission probability (= SAR)")
    ax.set_title("Regime B — endogenous feed-loop phase diagram\n"
                 f"ranking $H$: recency half-life {half_life:.0f}h (snapshot-fit) $+0.2\\cdot$upvotes  "
                 "| cyan = reliable per-model SAR band")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("final meme prevalence (adopters / N)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")          # PDF only (vector, for LaTeX)
    plt.close(fig)
    print(f"wrote {path}")


def replot_from_csv(csv=os.path.join(OUT, "phase_grid.csv")):
    df = pd.read_csv(csv)
    p_ret_grid = np.sort(df["p_ret"].unique())
    p_vote_grid = np.sort(df["p_vote"].unique())
    m_ret, m_vote = measured_point()
    _plot(df, p_ret_grid, p_vote_grid, m_ret, m_vote,
          os.path.join(FIGDIR, "fig_endogenous_phase.pdf"), half_life=fitted_half_life())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--N", type=int, default=5000)
    ap.add_argument("--cycles", type=int, default=120)
    ap.add_argument("--plot-only", action="store_true",
                    help="re-render the heatmap from out/endogenous_sim/phase_grid.csv")
    args = ap.parse_args()
    if args.plot_only:
        replot_from_csv()
    else:
        run(smoke=args.smoke, seed=args.seed, N=args.N, cycles=args.cycles)
