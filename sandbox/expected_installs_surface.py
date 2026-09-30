"""Expected installs + amplification across feed rankings — unified surface x cascade model.

MODEL VERSION: docs/SIMULATION_MODEL.md v2 (surface x cascade, pinned P(post)=0.10;
audited by docs/audit_simulation_model.py).

Replaces sandbox/expected_installs_table.py (and the installs half of amplification_combined).

MODEL (see memory surface-cascade-hybrid + docs/SIMULATION_MODEL.md). Per seeded Trojan attack
(model m, carrier c, feed ranking s):
  * the attacker's SEED post catches fire onto broadcast surface s with probability
        p_break(s, c) = P(peak_upvotes * r_up_c >= tau_s) * timing_s          [entry-bar model]
    where r_up_c = reaction(carrier c)/reaction(generic) is the carrier's upvote edge and
    (tau_s, timing_s) are surface s's calibrated entry bar + occupancy discount
    (analysis.hot_entry_probability). Carrier and surface enter ONLY through this scalar.
  * conditional on catching fire the payload holds the top feed slot and the cascade saturates
    to ~N*install (the viral outcome, broadcast pool); otherwise it stays buried and spreads
    only via the subcritical retransmission branching from the cold-start foothold (buried pool).
    endogenous_sim.simulate_surface_cascade generates both; the buried limit reproduces the
    earned-climb cascade numerically. Long tails are the ACTUAL p5/p95 of this mixture.

So per model there are just TWO Monte-Carlo pools (buried, broadcast); each (carrier, surface)
cell is a Bernoulli(p_break) mixture of them, and amplification is expressed relative to the
generic link-sharer's EXPECTED installs on the same surface (r_up = 1):
        amp_draw = installs_Trojan_draw / E[installs_generic]   (mean = E[inst_T]/E[inst_G]).

Three tables:
  * Table 1  (tab_expinst_headline)  : per feed ranking, the MEAN over models of each model's
    BEST carrier ("mean of best"), installs & amp each with (p5, p95).
  * Table A  (tab_expinst_bymodel)   : robustness by model — each model's best carrier per ranking.
  * Table B  (tab_expinst_bycarrier) : robustness by carrier — averaged over models, at a
    representative ranking (the surface shape is identical across carriers, just scaled).

All five backends are assessable: P(post) is pinned at P_POST_STATE (a shared volume constant),
so the per-model retransmission SAR = P_POST_STATE * P(payload|post, security) comes entirely from
the consistently prefill-measured payload transmission (no dedicated per-model SAR sweep needed).

    PYTHONPATH=. python -m sandbox.expected_installs_surface

Outputs: out/attack_reach/expected_installs_surface.csv, figures/tab_expinst_{headline,bymodel,bycarrier}.tex
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from analysis.hot_entry_probability import calibration
from sandbox.endogenous_sim import simulate_surface_cascade, CONFIG
from sandbox.reach_by_model import _pcurves_with_payload
from sandbox.ccdf_trojan_baseline import install_rank0
from sandbox import expected_installs_edge as edge

OUT = "out/attack_reach"
FIG = "figures"
N = CONFIG.N

# Per-encounter retransmission SAR = P(post) x P(payload|post). P(post) is a shared VOLUME constant
# (not a competition variable), so we PIN it at P_POST_STATE (matching the edge model's P_POST_REF)
# and let the per-model signal come entirely from the prefill-measured P(payload|post) of the
# security carrier. This is consistent across all models (no dedicated SAR sweep needed), so all five
# backends are assessable. sar(model) = P_POST_STATE * p_payload_given_post(model, security).
P_POST_STATE = 0.058    # mean empirical clean-feed p(post) over the 5 assay models (was 0.10 pinned); sensitivity in appendix
# Per-generation retransmission attenuation from the rephrasing assay (app:rephrasing): a retransmitted
# post is one more rephrasing removed from the pristine seed, so p_ret decays by RET_DECAY each cascade
# generation. Raw mean of the per-backend per-hop retention over all five backends (increases kept, not
# clipped): gpt-oss 0.64, qwen 0.81, command-r 0.92, deepseek 1.17, gemma2 1.10 -> mean 0.93. 1.0 = off.
RET_DECAY = 0.93
MODELS = [("gpt-oss-120B", "gptoss120b"), ("qwen-32B", "qwen32b"),
          ("deepseek-v4.1-flash", "deepseekflash"), ("gemma2-27B", "gemma2_27b"),
          ("command-r-35B", "commandr_35b")]


def _p_payload(tag, cell="sec_mg"):
    p = os.path.join("out/exposure", f"cross_child_{tag}_{cell}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p)); d = d[0] if isinstance(d, list) and d else d
    v = d.get("p_payload_given_post")
    return float(v) if isinstance(v, (int, float)) else None
CARRIERS = {"security": "sec_mg", "claw": "claw_cv", "shell": "shell_sf", "oclaw": "oclaw_ok",
            "karma": "karma_ke", "molt": "molt_mt", "econ": "econ_ch", "auton": "auton_sc",
            "consc": "consc_ss", "alpha": "alpha_af"}   # each carrier -> its OWN bespoke payload (2026-09-26)
SURFACES = [("top_hour", "top/hour"), ("top_day", "top/day"),
            ("top_week", "top/week"), ("hot", "hot")]
REP_SURFACE = "top_day"           # representative ranking for the compact by-carrier panel
# Carrier for the upvote-edge x retransmission decompositions. Security is the carrier the
# retransmission SAR is measured on, so both levers come from the same stimulus, and a fixed
# carrier avoids the winner's-curse bias of picking the max-r_up carrier per model.
DECOMP_CARRIER = "security"

# The upvote-edge x retransmission decomposition is emitted for two response metrics: INSTALLS
# (install-rate x reach; the *_inst pools) and EXPOSURE (install-rate-free audience; the *_r pools).
# Exposure isolates spread from per-model install propensity, so it is comparable across backends.
_MET = {
    "installs": dict(suf="", noun="installs", short="E[installs]", ylab="expected installs per seeded post",
                     csv="expected_installs_decomp", tab="tab_expinst_decomp", pdf="expinst_decomposition",
                     csv_bm="expected_installs_decomp_bymodel", tab_bm="tab_expinst_decomp_bymodel",
                     pdf_bm="expinst_decomp_bymodel"),
    "exposure": dict(suf="_r", noun="exposures", short="E[exposed]", ylab="expected agents exposed per seeded post",
                     csv="expected_exposure_decomp", tab="tab_expexp_decomp", pdf="expexp_decomposition",
                     csv_bm="expected_exposure_decomp_bymodel", tab_bm="tab_expexp_decomp_bymodel",
                     pdf_bm="expexp_decomp_bymodel"),
}

# --- Measured upvote edge (2026-09-26): r_up = P(upvote carrier post) / P(upvote a random link-free
# corpus post), both at the stimulus slot (data/upvote-edge-2026-09). Replaces the reaction_frac proxy.
UPVOTE_DIR = "data/upvote-edge-2026-09"
UPVOTE_TAG = {"gptoss120b": "gptoss120b", "deepseekflash": "deepseek41flash",
              "qwen32b": "qwen32b", "gemma2_27b": "gemma2_27b", "commandr_35b": "commandr_35b"}
# carrier -> upvote cell name (bespoke carriers use the short name; the rest match the crossing cell)
UPVOTE_CELL = {"security": "sec_mg", "claw": "claw", "shell": "shell", "oclaw": "oclaw",
               "karma": "karma", "molt": "molt_mt", "econ": "econ_ch", "auton": "auton_sc",
               "consc": "consc_ss", "alpha": "alpha_af"}


def _upvote_rup(tag, carrier):
    """Measured upvote edge for (model tag, carrier), or None if not collected yet."""
    ut = UPVOTE_TAG.get(tag, tag); cell = UPVOTE_CELL.get(carrier)
    fc = os.path.join(UPVOTE_DIR, f"upvote_{ut}_{cell}.json")
    fg = os.path.join(UPVOTE_DIR, f"upvote_{ut}_generic.json")
    if not (cell and os.path.exists(fc) and os.path.exists(fg)):
        return None
    pc = json.load(open(fc)).get("p_upvote_pay"); pg = json.load(open(fg)).get("p_upvote_pay")
    return (pc / pg) if (pc is not None and pg) else None


def _p_payload_generic(tag):
    """P(payload|post) of the generic link-sharer (cross_bare_<tag>_gen) — the generic baseline's SAR."""
    p = os.path.join("out/exposure", f"cross_bare_{tag}_gen.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p)); d = d[0] if isinstance(d, list) and d else d
    v = d.get("p_payload_given_post")
    return float(v) if isinstance(v, (int, float)) else None


M_POOL = 6000                     # cascade runs per (model, mode)
M_MIX = 60000                     # mixture draws per cell (for smooth p5/p95)


def _reaction(fname):
    p = os.path.join("out/exposure", fname)
    if not os.path.exists(p):
        return None
    d = json.load(open(p)); d = d[0] if isinstance(d, list) and d else d
    v = d.get("reaction_frac")
    return v if isinstance(v, (int, float)) else None


# Confirmatory re-estimates (data/confirmatory-bestcarrier-2026-09): each backend's best carrier was
# SELECTED on the exploratory crossing (n_cond=32) and re-measured on fresh, independent samples, so
# its r_up is free of the winner's-curse bias. The generic floor (the r_up denominator) is never
# selected on, so both runs are pooled (n-weighted) for precision.
N_EXPLO = 32
CONFIRM_N_GEN = {"gptoss120b": 128, "deepseekflash": 128,
                 "qwen32b": 64, "gemma2_27b": 64, "commandr_35b": 64}


def r_up_explo(tag, carrier):
    """Exploratory-only r_up. Used ONLY to select each model's best carrier (selection is frozen on
    the exploratory data; re-selecting after swapping in confirmatory numbers would re-introduce
    selection bias)."""
    pvG = _reaction(f"cross_bare_{tag}_gen.json")
    pvT = _reaction(f"cross_child_{tag}_{CARRIERS[carrier]}.json")
    if pvG is None or pvT is None:
        return None
    return pvT / max(pvG, 1e-9)


def _generic_floor(tag):
    """n-weighted pool of the exploratory and (if present) confirmatory generic-floor reaction rate."""
    pe = _reaction(f"cross_bare_{tag}_gen.json")
    pc = _reaction(f"confirm_bare_{tag}_gen.json")
    if pe is None:
        return pc
    if pc is None:
        return pe
    nc = CONFIRM_N_GEN[tag]
    return (N_EXPLO * pe + nc * pc) / (N_EXPLO + nc)


def r_up(tag, carrier):
    """Estimation r_up: the confirmatory numerator where one exists (the selected best carrier),
    else the exploratory one; over the pooled generic floor. Reduces to r_up_explo when no
    confirmatory files are present."""
    pvG = _generic_floor(tag)
    cell = CARRIERS[carrier]
    pvT = _reaction(f"confirm_child_{tag}_{cell}.json")
    if pvT is None:
        pvT = _reaction(f"cross_child_{tag}_{cell}.json")
    if pvG is None or pvT is None:
        return None
    return pvT / max(pvG, 1e-9)


def p_break(surface, mult):
    peak, tau, timing = calibration(surface)
    return min(1.0, float((peak * mult >= tau).mean()) * timing)


def _pools(seecurves, sar, install, seed=0, ret_decay=1.0):
    """Outcome pools for one model at retransmission rate `sar`, as
    (buried_inst, bcast_inst, buried_reach, bcast_reach). The *_inst arrays are expected installs
    (= install_rate x reach) and drive the E[installs] columns; the *_reach arrays are the exposed
    audience (install-rate-free) and drive the EXPOSURE amplification (so a model that never installs
    still has a well-defined amplification, and amp does not depend on the install rate at all).
    Call with sar=0 for the non-transmitting baseline (seed's own reach only, no cascade).
    ret_decay<1 attenuates retransmission per cascade generation (rephrasing-lossiness sensitivity)."""
    rng = np.random.default_rng(seed)
    b = [simulate_surface_cascade(sar, install, False, seecurves, rng, ret_decay=ret_decay) for _ in range(M_POOL)]
    c = [simulate_surface_cascade(sar, install, True, seecurves, rng, ret_decay=ret_decay) for _ in range(M_POOL)]
    bi = np.array([x[1] for x in b]); ci = np.array([x[1] for x in c])   # installs
    br = np.array([x[0] for x in b]); cr = np.array([x[0] for x in c])   # reach (exposure)
    return bi, ci, br, cr


def _mix(buried, bcast, pb, rng):
    """M_MIX installs draws from the Bernoulli(pb) mixture of the two pools."""
    fire = rng.random(M_MIX) < pb
    out = np.where(fire, rng.choice(bcast, M_MIX), rng.choice(buried, M_MIX))
    return out


NBOOT = 2000


def _stats(a):
    p5, p50, p95, p99 = np.percentile(a, [5, 50, 95, 99])
    return dict(mean=float(a.mean()), p5=float(p5), p50=float(p50),
                p95=float(p95), p99=float(p99))


def _bsm(a, rng, nboot):
    """bootstrap distribution of the MEAN of `a` (nboot resample-means)."""
    n = len(a)
    return a[rng.integers(0, n, (nboot, n))].mean(axis=1)


def _cell_ci(contribs, surface, seed=0, nboot=NBOOT):
    """Pool-level bootstrap 95% CI for a state cell (mean over model contributions).

    E[installs] columns (inst, gbase = generic, mbase = market-tip control) are means over models.
    Amplifications are EXPOSURE ratios computed as RATIO-OF-MEANS -- aggregate best-carrier reach over
    aggregate baseline reach across the model population -- not a mean of per-model ratios. This is
    robust when a baseline's reach is ~0 for some backend (the market-tip control never breaks in for
    deepseek, r_up=0), which would make a per-model ratio blow up. Returns dict(inst, gbase, mbase,
    amp, amp_mkt) each = (mean, lo, hi)."""
    rng = np.random.default_rng(seed)
    pb1 = p_break(surface, 1.0)
    K = len(contribs)
    I = np.zeros(nboot); G = np.zeros(nboot); Mk = np.zeros(nboot)      # E[installs] means
    RB = np.zeros(nboot); RG = np.zeros(nboot); RM = np.zeros(nboot)    # summed reach (ratio-of-means)
    inst_pt = gb_pt = mb_pt = 0.0
    rb_pt = rg_pt = rm_pt = 0.0
    Km = 0                                     # count of contribs that carry the market-tip control
    for P in contribs:                        # each P carries its carrier pools + model baselines + ru
        ru = P["ru"]; pbT = p_break(surface, ru)
        # E[installs] columns: from the install-outcome pools (install_rate x reach)
        bc, bu = _bsm(P["bcast"], rng, nboot), _bsm(P["buried"], rng, nboot)   # carrier SAR
        bcG, buG = _bsm(P["bcG"], rng, nboot), _bsm(P["buG"], rng, nboot)      # generic payload SAR
        inst_c = pbT * bc + (1 - pbT) * bu
        gb_c = pb1 * bcG + (1 - pb1) * buG            # generic post: generic payload, r_up=1
        I += inst_c / K; G += gb_c / K
        # exposure (reach) pools -> ratio-of-means amplification
        rbc, rbu = _bsm(P["bcast_r"], rng, nboot), _bsm(P["buried_r"], rng, nboot)
        rbcG, rbuG = _bsm(P["bcG_r"], rng, nboot), _bsm(P["buG_r"], rng, nboot)
        RB += pbT * rbc + (1 - pbT) * rbu
        RG += pb1 * rbcG + (1 - pb1) * rbuG
        inst_pt += (pbT * P["bcast"].mean() + (1 - pbT) * P["buried"].mean()) / K
        gb_pt += (pb1 * P["bcG"].mean() + (1 - pb1) * P["buG"].mean()) / K
        rb_pt += pbT * P["bcast_r"].mean() + (1 - pbT) * P["buried_r"].mean()
        rg_pt += pb1 * P["bcG_r"].mean() + (1 - pb1) * P["buG_r"].mean()
        # market-tip control (alpha carrier: its own r_up + SAR) -- a weak, pre-flagged-inert real
        # contagion. Absolute column (mbase) + a ratio-of-means amplification denominator (amp_mkt).
        if P.get("alpha_bu") is not None:
            pbM = p_break(surface, P["alpha_ru"]); Km += 1
            mbc, mbu = _bsm(P["alpha_bc"], rng, nboot), _bsm(P["alpha_bu"], rng, nboot)
            Mk += pbM * mbc + (1 - pbM) * mbu
            rmbc, rmbu = _bsm(P["alpha_bcr"], rng, nboot), _bsm(P["alpha_bur"], rng, nboot)
            RM += pbM * rmbc + (1 - pbM) * rmbu
            mb_pt += pbM * P["alpha_bc"].mean() + (1 - pbM) * P["alpha_bu"].mean()
            rm_pt += pbM * P["alpha_bcr"].mean() + (1 - pbM) * P["alpha_bur"].mean()
    ci = lambda pt, arr: (float(pt), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
    mbase = ci(mb_pt / Km, Mk / Km) if Km else (float("nan"),) * 3
    amp = ci(rb_pt / max(rg_pt, 1e-9), RB / np.maximum(RG, 1e-9))
    amp_mkt = ci(rb_pt / max(rm_pt, 1e-9), RB / np.maximum(RM, 1e-9)) if Km else (float("nan"),) * 3
    return dict(inst=ci(inst_pt, I), gbase=ci(gb_pt, G), mbase=mbase, amp=amp, amp_mkt=amp_mkt)


def build(seed=0, edge_reps=3000, ret_decay=RET_DECAY):
    seecurves = None    # state cascade uses explicit CONFIG.bcast_frac / coldstart_frac (no gpt-4o-mini)
    rng = np.random.default_rng(1234)
    print(f"retransmission per-generation decay ret_decay={ret_decay} (rephrasing attenuation; app:rephrasing)")
    # Only models whose measured upvote edge (security) is collected. Missing backends are simply
    # absent from the tables (placeholder regeneration until their upvote runs finish).
    models = [(n, t) for (n, t) in MODELS if _upvote_rup(t, "security") is not None]
    print("models with measured upvote data: " + ", ".join(n for n, _ in models))

    # Per-model pools: retransmission SAR is CARRIER-SPECIFIC (sar_c = P(post)*P(payload|post,carrier)).
    # Plus two model-level baselines: generic payload (r_up=1) and retransmission-off (SAR=0).
    pools = {}
    for name, tag in models:
        install = install_rank0(tag)
        pg = _p_payload_generic(tag) or 0.0
        buG, bcG, buG_r, bcG_r = _pools(seecurves, P_POST_STATE * pg, install, seed=seed + 2, ret_decay=ret_decay)  # generic payload
        bur0, bc0, bur0_r, bc0_r = _pools(seecurves, 0.0, install, seed=seed + 1)              # retransmission OFF
        car = {}
        for carrier, cell in CARRIERS.items():
            ru = _upvote_rup(tag, carrier); pc = _p_payload(tag, cell)
            if ru is None or pc is None:
                continue                      # carrier not collected for this model yet
            bu, bc, bu_r, bc_r = _pools(seecurves, P_POST_STATE * pc, install, seed=seed, ret_decay=ret_decay)
            car[carrier] = dict(buried=bu, bcast=bc, buried_r=bu_r, bcast_r=bc_r,
                                ru=ru, sar=P_POST_STATE * pc)
        sec = car.get("security")
        eG = {s: p_break(s, 1.0) * bcG.mean() + (1 - p_break(s, 1.0)) * buG.mean() for s, _ in SURFACES}
        pools[name] = dict(tag=tag, install=install, bur0=bur0, bc0=bc0, buG=buG, bcG=bcG, eG=eG,
                           bur0_r=bur0_r, bc0_r=bc0_r, buG_r=buG_r, bcG_r=bcG_r,
                           car=car, buried=(sec and sec["buried"]), bcast=(sec and sec["bcast"]))

    # ---- per (model, carrier, surface) installs + amp draws --------------------------------
    rows = []
    draws = {}     # (model, carrier, surface) -> (installs, amp, gbase, nbase) draws
    rng_g = np.random.default_rng(4321)   # dedicated stream for the generic-link CCDF baseline draw,
    # so adding it does not perturb the main `rng` sequence (keeps all other outputs byte-stable)
    for name, _ in models:
        P = pools[name]
        for carrier, C in P["car"].items():
            ru = C["ru"]
            for skey, slabel in SURFACES:
                pb = p_break(skey, ru)
                inst = _mix(C["buried"], C["bcast"], pb, rng)
                #  gbase = generic post (generic payload, r_up=1)  -> gap = upvote-edge contribution.
                #  nbase = this carrier break-in but retransmission OFF -> gap = retransmission contrib.
                gbase = _mix(P["buG"], P["bcG"], p_break(skey, 1.0), rng)
                nbase = _mix(P["bur0"], P["bc0"], pb, rng)
                amp = inst / max(P["eG"][skey], 1e-9)
                # exposure (reach) draws: install-rate-free, for the exposure CCDF
                reach = _mix(C["buried_r"], C["bcast_r"], pb, rng)
                reach_nb = _mix(P["bur0_r"], P["bc0_r"], pb, rng)
                # market-tip control draws (alpha carrier: own r_up + SAR), kept for reference
                A = P["car"].get("alpha")
                if A is not None:
                    pbM = p_break(skey, A["ru"])
                    mbase = _mix(A["buried"], A["bcast"], pbM, rng)
                    reach_mkt = _mix(A["buried_r"], A["bcast_r"], pbM, rng)
                else:
                    mbase, reach_mkt = nbase, reach_nb
                # generic-link reach (CCDF dashed baseline) -- drawn from a DEDICATED rng so adding it
                # leaves the main rng sequence (and thus every other output) byte-identical.
                reach_g = _mix(P["buG_r"], P["bcG_r"], p_break(skey, 1.0), rng_g)
                draws[(name, carrier, skey)] = (inst, amp, gbase, nbase, reach, reach_nb, mbase, reach_mkt, reach_g)
                si, sa, sg, sn = _stats(inst), _stats(amp), _stats(gbase), _stats(nbase)
                rows.append(dict(model=name, carrier=carrier, r_up=round(ru, 3),
                                 surface=slabel, p_break=round(pb, 4),
                                 inst_mean=si["mean"], inst_p50=si["p50"], inst_p95=si["p95"],
                                 inst_p99=si["p99"],
                                 gbase_mean=sg["mean"], gbase_p95=sg["p95"], gbase_p99=sg["p99"],
                                 nbase_mean=sn["mean"], nbase_p95=sn["p95"], nbase_p99=sn["p99"],
                                 amp_mean=sa["mean"], amp_p50=sa["p50"], amp_p95=sa["p95"],
                                 amp_p99=sa["p99"]))
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "expected_installs_surface.csv"), index=False)

    # best carrier per model = max expected installs at the representative ranking (reflects BOTH
    # the upvote edge and the carrier-specific retransmission rate).
    best = {}
    for name, _ in models:
        P = pools[name]
        if not P["car"]:
            best[name] = None; continue
        def _etd(c):
            C = P["car"][c]; pb = p_break(REP_SURFACE, C["ru"])
            return pb * C["bcast"].mean() + (1 - pb) * C["buried"].mean()
        best[name] = max(P["car"], key=_etd)
    print("best carrier per model: " + ", ".join(f"{m}={best[m]}" for m, _ in models))

    # pool-level bootstrap 95% CIs for the table cells (the CCDF figure owns the outcome tail)
    def contribs(model_carriers):
        out = []
        for m, c in model_carriers:
            P = pools.get(m); C = P and P["car"].get(c)
            if C is None:
                continue
            A = P["car"].get("alpha")          # market-tip control carrier for this model
            out.append(dict(buried=C["buried"], bcast=C["bcast"], ru=C["ru"],
                            bur0=P["bur0"], bc0=P["bc0"], buG=P["buG"], bcG=P["bcG"],
                            buried_r=C["buried_r"], bcast_r=C["bcast_r"],
                            bur0_r=P["bur0_r"], bc0_r=P["bc0_r"], buG_r=P["buG_r"], bcG_r=P["bcG_r"],
                            alpha_bu=(A and A["buried"]), alpha_bc=(A and A["bcast"]),
                            alpha_bur=(A and A["buried_r"]), alpha_bcr=(A and A["bcast_r"]),
                            alpha_ru=(A and A["ru"])))
        return out
    hl_ci = {skey: _cell_ci(contribs([(m, best[m]) for m, _ in models if best[m]]), skey)
             for skey, _ in SURFACES}
    bm_ci = {(m, skey): _cell_ci(contribs([(m, best[m])]), skey)
             for m, _ in models if best[m] for skey, _ in SURFACES}
    bc_ci = {}
    for c in CARRIERS:
        cc = contribs([(m, c) for m, _ in models])
        if cc:
            bc_ci[c] = (len(cc), _cell_ci(cc, REP_SURFACE))

    # edge-mediated results (follower graph), folded into the headline + by-model tables as
    # subtables; the by-carrier table stays state-only (the edge cascade is carrier-independent).
    print(f"\ncomputing EDGE-mediated results ({edge_reps} cascades/cell) for the subtables ...")
    eres = edge.compute(reps=edge_reps)
    _table_headline(hl_ci, eres)
    _table_bymodel(bm_ci, best, eres)
    _table_bycarrier(bc_ci)
    for metric in ("installs", "exposure"):   # same factorial on installs and on exposure (reach)
        _decomposition(pools, best, metric)          # effect sizes: upvote-edge vs retransmission
        _decomposition_by_model(pools, best, metric) # per-model version (grouped bars)
    _ccdf_figure(draws, best, eres, "installs")   # install exceedance CCDFs (state | edge)
    _ccdf_figure(draws, best, eres, "exposure")   # exposure (reach) exceedance CCDFs -- cleaner instrument
    print(f"\nwrote {OUT}/expected_installs_surface.csv, {OUT}/expected_installs_edge.csv and "
          f"{FIG}/tab_expinst_{{headline,bymodel,bycarrier,decomp,decomp_bymodel}}.tex, "
          f"{FIG}/tab_expexp_decomp{{,_bymodel}}.tex, {FIG}/expinst_decomposition.pdf, "
          f"{FIG}/expexp_decomposition.pdf, {FIG}/expinst_decomp_bymodel.pdf, "
          f"{FIG}/expexp_decomp_bymodel.pdf, {FIG}/expinst_ccdf.pdf, {FIG}/expexp_ccdf.pdf")


# ------------------------------------------------------------------ console + tex helpers ----
def _fmt(mean, lo, hi, d=0):
    return f"{mean:,.{d}f} [{lo:,.{d}f}, {hi:,.{d}f}]".replace(",", "{,}")




_HDR5 = ("E[installs] & generic & market-tip & Exp.Amp$_{\\text{gen}}\\times$ & Exp.Amp$_{\\text{mkt}}\\times$\\\\")


def _section(ncol, text):
    return f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\emph{{{text}}}}}\\\\"


def _state_rows_headline(hl_ci):
    print("\n=== Table 1a STATE (mean of best over models, 95% CI) ===")
    print(f"  {'ranking':9}{'E[installs]':>22}{'generic base':>22}{'market-tip base':>22}"
          f"{'amp/gen':>10}{'amp/mkt':>10}")
    out = []
    for skey, slabel in SURFACES:
        d = hl_ci[skey]
        out.append(f"\\texttt{{{slabel}}} & {_fmt(*d['inst'])} & {_fmt(*d['gbase'])} & "
                   f"{_fmt(*d['mbase'])} & {_fmt(*d['amp'], d=2)} & {_fmt(*d['amp_mkt'], d=2)}\\\\")
        f0 = lambda t: f"{t[0]:,.0f}[{t[1]:,.0f},{t[2]:,.0f}]"
        print(f"  {slabel:9}{f0(d['inst']):>22}{f0(d['gbase']):>22}{f0(d['mbase']):>22}"
              f"{d['amp'][0]:>10.2f}{d['amp_mkt'][0]:>10.2f}")
    return out


def _table_headline(hl_ci, eres):
    """Merged Table 1: one tabular, state and edge rows delineated by a multicolumn section header."""
    lines = ["% Table 1 (merged) -- expected installs + amplification, mean over models, both",
             "% exposure mechanisms in one tabular. Generated by sandbox.expected_installs_surface.",
             "% Needs booktabs.",
             "\\begin{table*}[t]\\centering\\small",
             "\\caption{\\textbf{Expected payload installs and amplification, per seeded Trojan "
             "post.} Mean over models; brackets are bootstrap 95\\% CIs of the mean (the outcome "
             "distribution is the companion install-exceedance CCDF figure). \\emph{generic} $=$ a "
             "generic link-sharer (an ordinary link post, $r_{up}{=}1$); \\emph{market-tip} $=$ a "
             "weak, pre-flagged-inert real contagion (fails the $\\phi_e$ transmission placebo and "
             "$r_{up}{<}1$), an ecological negative control. E[installs], generic and market-tip are "
             "absolute expected installs (install characterization). Exp.Amp$_{\\text{gen}}$/"
             "Exp.Amp$_{\\text{mkt}}$ are the \\emph{exposure} (audience reached) of the best carrier "
             "relative to the generic / market-tip baseline, as a ratio of mean reach over the model "
             "population (install-rate-free, and robust to a baseline whose reach is $\\approx0$ for "
             "some backend). Each model at its best carrier; the "
             "carrier-specific retransmission rate is $P(\\text{post}){=}0.058$ (mean empirical) "
             "$\\times$ the measured $P(\\text{payload}\\mid\\text{post})$ of that carrier, and the "
             "upvote edge $r_{up}$ is the measured upvote rate relative to a random link-free post. "
             "Edge: models pooled (no upvote channel, so security-Trojan vs generic).}",
             "\\label{tab:expinst-headline}",
             "\\begin{tabular}{@{}l r r r r r@{}}", "\\toprule",
             f"Feed ranking / seed & {_HDR5}", "\\midrule",
             _section(6, "State-mediated feed broadcast (Moltbook $N{=}37{,}189$; by feed ranking)"),
             *_state_rows_headline(hl_ci), "\\addlinespace",
             _section(6, "Edge-mediated follower cascade (SNAP $N{=}81{,}306$; by seed follower count)"),
             *edge.rows_headline(eres),
             "\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    open(os.path.join(FIG, "tab_expinst_headline.tex"), "w").write("\n".join(lines))


def _state_rows_bymodel(bm_ci, best):
    out = []
    for name, tag in MODELS:
        c = best.get(name)
        if not c:
            continue                          # model without measured upvote data (placeholder run)
        ru = _upvote_rup(tag, c)
        first = True
        for skey, slabel in SURFACES:
            d = bm_ci.get((name, skey))
            if d is None:
                continue
            mcell = (f"\\texttt{{{name}}} ({c}, $r_{{up}}{{=}}{ru:.2f}$)" if first else "")
            out.append(f"{mcell} & \\texttt{{{slabel}}} & "
                       f"{_fmt(*d['inst'])} & {_fmt(*d['gbase'])} & "
                       f"{_fmt(*d['mbase'])} & {_fmt(*d['amp'], d=2)} & {_fmt(*d['amp_mkt'], d=2)}\\\\")
            first = False
        out.append("\\addlinespace")
    return out


def _table_bymodel(bm_ci, best, eres):
    """Merged Table A: one tabular, state and edge rows delineated by a multicolumn section header."""
    lines = ["% Table A (merged) -- robustness by model, both exposure mechanisms in one tabular.",
             "% Generated by sandbox.expected_installs_surface. Needs booktabs + rotating.",
             "\\begin{sidewaystable*}\\centering\\small",
             "\\caption{\\textbf{Robustness by model} (Table~\\ref{tab:expinst-headline} broken "
             "out), both exposure mechanisms. Absolute installs with the two baselines (generic link, "
             "market-tip control) and the two \\emph{exposure} amplifications over them (audience "
             "reached, install-rate-free), each mean with a bootstrap 95\\% CI. State: each model's best "
             "carrier ($r_{up}$, the measured upvote edge) across feed rankings; the "
             "carrier-specific retransmission rate is $P(\\text{post}){=}0.058$ (mean empirical) "
             "$\\times$ the measured $P(\\text{payload}\\mid\\text{post})$ of that carrier.}",
             "\\label{tab:expinst-bymodel}",
             "\\begin{tabular}{@{}l l r r r r r@{}}", "\\toprule",
             f"Model & ranking / seed & {_HDR5}", "\\midrule",
             _section(7, "State-mediated feed broadcast (by feed ranking, best carrier per model)"),
             *_state_rows_bymodel(bm_ci, best),
             _section(7, "Edge-mediated follower cascade (by seed follower count)"),
             *edge.rows_bymodel(eres),
             "\\bottomrule", "\\end{tabular}", "\\end{sidewaystable*}", ""]
    open(os.path.join(FIG, "tab_expinst_bymodel.tex"), "w").write("\n".join(lines))


def _table_bycarrier(bc_ci):
    """Compact by-carrier panel at the representative ranking, averaged over models."""
    slabel = dict(SURFACES)[REP_SURFACE]
    rec = [(c, nmod, d) for c, (nmod, d) in bc_ci.items()]
    rec.sort(key=lambda t: -t[2]["amp"][0])       # rank by amplification
    lines = ["% Table B -- robustness by carrier (meme), averaged over models, at the",
             f"% representative ranking ({slabel}). Generated by sandbox.expected_installs_surface.",
             "% Needs booktabs + rotating.",
             "\\begin{sidewaystable*}\\centering\\small",
             "\\caption{\\textbf{Robustness by carrier (meme).} Averaged over models, at the "
             f"\\texttt{{{slabel}}} ranking (the ranking shape is identical across carriers, just "
             "scaled by $p_{\\text{break}}$, so one ranking suffices). Installs, the \\emph{generic} "
             "baseline (upvote-edge null, $r_{up}{=}1$), and the exposure amplification over it, each "
             "mean with a bootstrap 95\\% CI; ranked by amplification. Every carrier was measured on "
             "all five backends. (The market-tip control is a fixed reference, so it is "
             "shown once in Tables~\\ref{tab:expinst-headline},~\\ref{tab:expinst-bymodel} rather than "
             "per carrier here.)}",
             "\\label{tab:expinst-bycarrier}",
             "\\begin{tabular}{@{}l r r r@{}}", "\\toprule",
             "Carrier & E[installs] & generic & Amp$_{\\text{gen}}\\times$\\\\", "\\midrule"]
    for c, nmod, d in rec:
        lines.append(f"\\texttt{{{c}}} & {_fmt(*d['inst'])} & {_fmt(*d['gbase'])} & "
                     f"{_fmt(*d['amp'], d=2)}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\vspace{2pt}\\par\\footnotesize State-mediated only: the edge-mediated "
              "(follower-graph) cascade has no karma/upvote channel, so it does not vary by carrier "
              "and has no by-carrier breakdown (see the edge rows of "
              "Tables~\\ref{tab:expinst-headline} and~\\ref{tab:expinst-bymodel}).",
              "\\end{sidewaystable*}", ""]
    open(os.path.join(FIG, "tab_expinst_bycarrier.tex"), "w").write("\n".join(lines))


def _decomposition(pools, best, metric="installs"):
    """2x2 factorial decomposition of E[installs] (or E[exposed]) into the UPVOTE-EDGE and RETRANSMISSION effects.
    Four corners per (model, surface) at DECOMP_CARRIER (r_up=ru): U in {generic r_up=1,
    trojan r_up=ru} x R in {retransmission off, on}, using the analytic mixture mean
    E = p_break*bcast.mean() + (1-p_break)*buried.mean() (exact; no MC noise):
        n00 U0R0 = generic, no-retx   n10 U1R0 = trojan, no-retx (= non-tx base)
        n01 U0R1 = generic, retx (= generic base)   n11 U1R1 = full E[installs]
    Main effects (average over the other factor) + interaction, pooled (mean over models):
        dU = 1/2[(n10-n00)+(n11-n01)]   dR = 1/2[(n01-n00)+(n11-n10)]
        inter = n11-n10-n01+n00         total = n11-n00 = dU+dR
    (The averaged main effects already split the interaction equally between the two levers, so
    dU+dR sums exactly to the total lift; `inter` is reported separately as the non-additivity.)
    Writes figures/tab_expinst_decomp.tex + figures/expinst_decomposition.pdf (mean over the
    models at DECOMP_CARRIER)."""
    M = _MET[metric]; sf = M["suf"]
    rows = []
    for skey, slabel in SURFACES:
        acc = {k: [] for k in ("n00", "n10", "n01", "n11")}
        for name, P in pools.items():
            C = P["car"].get(best.get(name))     # each model's best carrier (the attacker's choice)
            if C is None:
                continue
            pbT, pb1 = p_break(skey, C["ru"]), p_break(skey, 1.0)
            bB, buB = C["bcast" + sf].mean(), C["buried" + sf].mean()   # carrier retransmission ON
            bB0, buB0 = P["bc0" + sf].mean(), P["bur0" + sf].mean()     # retransmission OFF
            acc["n00"].append(pb1 * bB0 + (1 - pb1) * buB0)          # generic, no-retx
            acc["n10"].append(pbT * bB0 + (1 - pbT) * buB0)          # trojan,  no-retx
            acc["n01"].append(pb1 * bB + (1 - pb1) * buB)            # upvote-off, retx on (same carrier payload)
            acc["n11"].append(pbT * bB + (1 - pbT) * buB)            # full response
        n00, n10, n01, n11 = (float(np.mean(acc[k])) for k in ("n00", "n10", "n01", "n11"))
        dU = 0.5 * ((n10 - n00) + (n11 - n01))
        dR = 0.5 * ((n01 - n00) + (n11 - n10))
        inter = n11 - n10 - n01 + n00
        rows.append(dict(surface=slabel, base=n00, dU=dU, dR=dR, inter=inter, total=n11 - n00,
                         value=n11))
    dec = pd.DataFrame(rows)
    dec.to_csv(os.path.join(OUT, f"{M['csv']}.csv"), index=False)
    print(f"\n=== Effect-size decomposition (mean over models, best carrier per model): "
          f"Delta {M['noun']} from upvote-edge vs retransmission ===")
    print(f"  {'ranking':9}{'base(null)':>12}{'dUpvote':>11}{'dRetx':>11}{'interact':>11}{'=resp':>11}")
    for r in rows:
        print(f"  {r['surface']:9}{r['base']:>12.0f}{r['dU']:>11.0f}{r['dR']:>11.0f}"
              f"{r['inter']:>11.0f}{r['value']:>11.0f}")
    _decomp_table(dec, metric)
    _decomp_plot(dec, metric)


def _decomp_table(dec, metric="installs"):
    M = _MET[metric]; short = M["short"]; lab = "tab:expexp-decomp" if metric == "exposure" else "tab:expinst-decomp"
    lines = [f"% Effect-size decomposition of {short} (state-mediated): upvote-edge vs",
             "% retransmission, 2x2 factorial main effects. sandbox.expected_installs_surface.",
             "\\begin{table}[t]\\centering\\small",
             f"\\caption{{\\textbf{{Effect sizes: upvote-edge vs retransmission on expected {M['noun']}}} "
             "(state-mediated, mean over models, security carrier). A "
             f"$2{{\\times}}2$ factorial split of {short} over a double null (generic post, no "
             "retransmission): $\\Delta$\\emph{upvote} is the main effect of the Trojan's upvote "
             "edge (break-in), $\\Delta$\\emph{retx} the main effect of retransmission, and "
             "the two main effects sum exactly to the total lift over the null, and \\emph{inter} "
             "(already apportioned equally between them) measures their non-additivity. The upvote "
             "edge governs feed break-in and the carrier-specific retransmission rate governs the "
             "post-break cascade; their relative size varies by feed ranking.}",
             f"\\label{{{lab}}}",
             "\\begin{tabular}{@{}l r r r r r@{}}", "\\toprule",
             f"Feed ranking & null base & $\\Delta$upvote & $\\Delta$retx & inter & $=${short}\\\\",
             "\\midrule"]
    for _, r in dec.iterrows():
        f = lambda v: f"{v:,.0f}".replace(",", "{,}")
        lines.append(f"\\texttt{{{r.surface}}} & {f(r.base)} & {f(r.dU)} & {f(r.dR)} & "
                     f"{f(r.inter)} & {f(r.value)}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    open(os.path.join(FIG, f"{M['tab']}.tex"), "w").write("\n".join(lines))


def _decomposition_by_model(pools, best, metric="installs"):
    """Per-model 2x2 factorial (upvote-edge x retransmission) at DECOMP_CARRIER, AVERAGED over
    the four feed rankings. Reveals whether the lever balance is model-specific (companion to the
    mean-over-models figure). Writes the CSV / tex / pdf named for `metric` (installs or exposure)."""
    M = _MET[metric]; sf = M["suf"]
    rows = []
    for name, P in pools.items():
        C = P["car"].get(best.get(name))         # each model's best carrier
        if C is None:
            continue
        ru = C["ru"]
        bB, buB = C["bcast" + sf].mean(), C["buried" + sf].mean()   # carrier retransmission ON
        bB0, buB0 = P["bc0" + sf].mean(), P["bur0" + sf].mean()     # retransmission OFF
        acc = {k: [] for k in ("base", "dU", "dR", "inter", "value")}
        for skey, _ in SURFACES:
            pbT, pb1 = p_break(skey, ru), p_break(skey, 1.0)
            n00 = pb1 * bB0 + (1 - pb1) * buB0                    # generic, no-retx (double null)
            n10 = pbT * bB0 + (1 - pbT) * buB0                    # trojan,  no-retx
            n01 = pb1 * bB + (1 - pb1) * buB                      # upvote-off, retx on (same carrier payload)
            n11 = pbT * bB + (1 - pbT) * buB                      # full response
            acc["base"].append(n00)
            acc["dU"].append(0.5 * ((n10 - n00) + (n11 - n01)))
            acc["dR"].append(0.5 * ((n01 - n00) + (n11 - n10)))
            acc["inter"].append(n11 - n10 - n01 + n00)
            acc["value"].append(n11)
        row = dict(model=name, carrier=best.get(name),
                   base=float(np.mean(acc["base"])), dU=float(np.mean(acc["dU"])),
                   dR=float(np.mean(acc["dR"])), inter=float(np.mean(acc["inter"])),
                   value=float(np.mean(acc["value"])))
        row["total"] = row["dU"] + row["dR"]           # = mean(n11 - n00); inter is not additive
        rows.append(row)
    dec = pd.DataFrame(rows)
    dec.to_csv(os.path.join(OUT, f"{M['csv_bm']}.csv"), index=False)
    print(f"\n=== Per-model {M['noun']} decomposition (mean over feed rankings, best carrier) ===")
    print(f"  {'model':20}{'carrier':10}{'dUpvote':>10}{'dRetx':>10}{'interact':>10}{'=resp':>10}")
    for r in rows:
        print(f"  {r['model']:20}{r['carrier']:10}{r['dU']:>10.0f}{r['dR']:>10.0f}"
              f"{r['inter']:>10.0f}{r['value']:>10.0f}")
    _decomp_bymodel_table(dec, metric)
    _decomp_bymodel_plot(dec, metric)


def _decomp_bymodel_table(dec, metric="installs"):
    M = _MET[metric]; short = M["short"]
    lab = "tab:expexp-decomp-bymodel" if metric == "exposure" else "tab:expinst-decomp-bymodel"
    f = lambda v: f"{v:,.0f}".replace(",", "{,}")
    lines = [
        f"% Per-model effect-size decomposition of {short} (state-mediated), 2x2 factorial",
        "% (upvote-edge x retransmission), mean over the four feed rankings, security carrier.",
        "% sandbox.expected_installs_surface._decomposition_by_model.",
        "\\begin{table}[t]\\centering\\small",
        f"\\caption{{\\textbf{{Per-model effect sizes: upvote-edge vs retransmission on expected "
        f"{M['noun']}}} (state-mediated, mean over the top/hour, top/day, top/week and hot rankings, "
        "\\texttt{security} carrier --- the carrier on which retransmission is measured). "
        f"$2{{\\times}}2$ factorial split of {short} over a double null (generic post, no "
        "retransmission): $\\Delta$\\emph{upvote} is the carrier's upvote-edge (break-in) main "
        "effect, $\\Delta$\\emph{retx} the retransmission main effect; the two sum exactly to the "
        "lift over the null, and \\emph{inter} (already apportioned equally between them) measures "
        "their non-additivity. \\emph{Upvote share} $=\\Delta$upvote$/(\\Delta$upvote$+\\Delta$retx). "
        "The balance between the two levers is backend-specific.}",
        f"\\label{{{lab}}}",
        "\\begin{tabular}{@{}l r r r r r@{}}", "\\toprule",
        f"Backend & $\\Delta$upvote & $\\Delta$retx & inter & upvote share & $=${short}\\\\",
        "\\midrule"]
    for _, r in dec.iterrows():
        share = 100 * r.dU / (r.dU + r.dR) if (r.dU + r.dR) else float("nan")
        lines.append(f"{r.model} & {f(r.dU)} & {f(r.dR)} & {f(r.inter)} & "
                     f"{share:.0f}\\% & {f(r.value)}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    open(os.path.join(FIG, f"{M['tab_bm']}.tex"), "w").write("\n".join(lines))


def _decomp_bymodel_plot(dec, metric="installs"):
    M = _MET[metric]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as _np
    effects = [("dU", "$\\Delta$ upvote-edge", "#e6821e"),
               ("dR", "$\\Delta$ retransmission", "#c0392b"),
               ("inter", "interaction (non-additivity)", "#4c78a8")]
    models = dec["model"].tolist()
    x = _np.arange(len(models)); w = 0.26
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for i, (col, lab, c) in enumerate(effects):
        vals = dec[col].to_numpy()
        bars = ax.bar(x + (i - 1) * w, vals, w, color=c, edgecolor="k",
                      linewidth=0.4, label=lab, zorder=3)
        ax.bar_label(bars, labels=[f"{v:,.0f}" for v in vals], padding=2, fontsize=6.0)
    # symlog y: the effects span ~300 (command-r) to ~6000 (gpt-oss); linear below `linthresh`
    # so the small negative interaction stays readable. Bar LENGTHS are log-compressed — read the
    # printed labels / the .tex table for exact magnitudes.
    ax.set_yscale("symlog", linthresh=100)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8)
    ax.set_ylabel(f"effect on {M['short']} per seeding (symlog)")
    ax.set_title(f"Per-model {M['noun']} decomposition", fontsize=11)
    ax.grid(alpha=0.25, axis="y", which="both")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8.5, ncol=3, loc="upper center", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"{M['pdf_bm']}.pdf"), bbox_inches="tight")
    plt.close(fig)


ROW_COL = ["#c0392b", "#e6821e", "#2a8f6b", "#4c78a8"]   # easiest/most-connected -> hardest/least


def _ccdf_curve(ax, a, xs, **kw):
    a = np.asarray(a, float)
    y = np.array([(a > x).mean() for x in xs])
    ax.plot(xs, np.where(y > 0, y, np.nan), **kw)


def _ccdf_figure(draws, best, eres, metric="installs"):
    """Companion to the headline Table 1 (same MEAN-OVER-MODELS aggregation): a single figure with
    two panels side by side, each an exceedance CCDF P(response > x) with ONE curve per headline row
    -- STATE-mediated per feed ranking (best carrier, pooled over models) | EDGE-mediated per
    seed-connectivity prototype (pooled over all models). Log-log. metric selects the response:
    'installs' (install-outcome draws) or 'exposure' (install-rate-free reach draws)."""
    exp = metric == "exposure"
    si, sb = (4, 8) if exp else (0, 2)               # STATE draw indices: (with-tx, generic-link baseline)
    eT, eB = ("Tr", "Gr") if exp else ("T", "G")     # EDGE draw keys (with-tx, generic link-sharer)
    noun = "exposures" if exp else "installs"
    verb = "exposed" if exp else "installed"
    pdf = "expexp_ccdf.pdf" if exp else "expinst_ccdf.pdf"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 9})
    fig, (axS, axE) = plt.subplots(1, 2, figsize=(11.4, 4.7))

    # --- (a) STATE: one curve per feed ranking, pooled over models (best carrier); dashed same
    #     colour = the generic link-sharer baseline for that ranking ---
    state_keys = {skey: [(m, best[m], skey) for m, _ in MODELS
                         if best.get(m) and (m, best[m], skey) in draws] for skey, _ in SURFACES}
    if exp:
        allv = [draws[k][si] for ks in state_keys.values() for k in ks]
        xmaxS = int(np.concatenate(allv).max()) if allv else 37189
    else:
        xmaxS = 37189                                # install-outcome ceiling (state population estimate)
    xsS = np.unique(np.round(np.logspace(0, np.log10(xmaxS) + 0.05, 130)).astype(int))
    for (skey, slabel), col in zip(SURFACES, ROW_COL):
        keys = state_keys[skey]
        if not keys:
            continue
        _ccdf_curve(axS, np.concatenate([draws[k][si] for k in keys]), xsS, color=col, lw=2, label=slabel)
        _ccdf_curve(axS, np.concatenate([draws[k][sb] for k in keys]), xsS, color=col, lw=1.2,
                    ls=(0, (4, 2)), alpha=0.9)                    # generic-link baseline
    axS.axvline(xmaxS, color="#2a7", ls="-.", lw=1.3)
    axS.text(xmaxS, 0.5, f"$N={xmaxS:,}$", fontsize=7.5, color="#178", rotation=90,
             ha="right", va="center", transform=axS.get_xaxis_transform())
    axS.set_title("State-mediated feed broadcast\n(mean over models, best carrier)", fontsize=10)

    # --- (b) EDGE: one curve per seed-connectivity prototype, pooled over models; dashed = generic ---
    xmaxE = eres["N"]
    xsE = np.unique(np.round(np.logspace(0, np.log10(xmaxE) + 0.05, 130)).astype(int))
    for (proto, plabel), col in zip(eres["PROTOS"], ROW_COL):
        keys = [(m, proto) for m, _ in edge.MODELS if (m, proto) in eres["raw"]]
        if not keys:
            continue
        flab = f"{proto} ({eres['foll'][proto]:,} foll.)"
        _ccdf_curve(axE, np.concatenate([eres["raw"][k][eT] for k in keys]), xsE, color=col, lw=2, label=flab)
        _ccdf_curve(axE, np.concatenate([eres["raw"][k][eB] for k in keys]), xsE, color=col, lw=1.2,
                    ls=(0, (4, 2)), alpha=0.9)                    # generic-link baseline
    axE.axvline(xmaxE, color="#2a7", ls="-.", lw=1.3)
    axE.text(xmaxE, 0.5, f"$N={xmaxE:,}$", fontsize=7.5, color="#178", rotation=90,
             ha="right", va="center", transform=axE.get_xaxis_transform())
    axE.set_title("Edge-mediated follower cascade\n(mean over models)", fontsize=10)

    from matplotlib.lines import Line2D
    style = [Line2D([0], [0], color="0.3", lw=2, label="with transmission"),
             Line2D([0], [0], color="0.3", lw=1.2, ls=(0, (4, 2)), label="generic link-sharer")]
    for ax, title in ((axS, "feed ranking"), (axE, "seed connectivity")):
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(f"x  (payload {noun})"); ax.set_ylabel(f"P({noun} > x)")
        ax.set_ylim(0.5 / 60000, 1.3); ax.grid(alpha=0.25, which="both")
        ax.spines[["top", "right"]].set_visible(False)
        ax.add_artist(ax.legend(fontsize=7.5, loc="lower left", title=title))   # colour = row
        ax.legend(handles=style, fontsize=7, loc="upper right", framealpha=0.9)  # linestyle key
    fig.suptitle(f"Payload-{verb} exceedance (CCDF) by exposure network — companion to Table 1",
                 fontsize=11, y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, pdf), bbox_inches="tight")
    plt.close(fig)


def _decomp_plot(dec, metric="installs"):
    M = _MET[metric]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 9})
    surf = list(dec.surface); x = np.arange(len(surf))
    base = dec.base.to_numpy(); dU = dec.dU.to_numpy(); dR = dec.dR.to_numpy(); inter = dec.inter.to_numpy()
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    b = base
    ax.bar(x, base, 0.62, color="#c2c6cc", edgecolor="black", lw=0.4, label="null base (generic, no retx)")
    ax.bar(x, dR, 0.62, bottom=b, color="#4c78a8", edgecolor="black", lw=0.4, label="$\\Delta$ retransmission"); b = b + dR
    ax.bar(x, dU, 0.62, bottom=b, color="#e45756", edgecolor="black", lw=0.4, label="$\\Delta$ upvote edge"); b = b + dU
    # base + dR + dU == full response exactly (the averaged main effects already absorb the
    # interaction), so the interaction is annotated, not stacked (stacking would double-count it).
    for xi, tot, it in zip(x, dec.value.to_numpy(), inter):
        ax.text(xi, tot, f"{tot:,.0f}\n(inter {it:+,.0f})", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(surf)
    ax.set_ylabel(M["ylab"])
    ax.set_title(f"Effect of upvote-edge vs retransmission on {M['noun']}\n"
                 "(state-mediated, mean over models, security carrier)", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"{M['pdf']}.pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ret-decay", type=float, default=RET_DECAY,
                    help="per-generation retransmission attenuation (1.0 = none; default = measured 0.82)")
    args = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    build(seed=args.seed, ret_decay=args.ret_decay)
