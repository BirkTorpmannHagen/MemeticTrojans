"""Per-carrier view of the TWO measured simulation inputs, as stacked facets.

Top facet -- transmission (retransmission channel):
  blue  = PURE contagion: the carrier idea seeded alone, NO link   cross_parent_<c> p_meme_semantic
          (re-transmission of the idea itself -- P(meme|post))
  color = memetic TROJAN: payload under the full carrier           cross_child_<c>_<pk> p_payload_given_post
          (payload transmission -- P(payload|post))
  The dotted line is the clean payload baseline: the true bare link (generic link-sharer with the
  description stripped, "check out this new skill", cross_bare_<tag>_gen).

Bottom facet -- upvote edge r_up (visibility channel, the other sim input):
  the carrier's measured upvote propensity relative to an ordinary Moltbook post
  (r_up = P(upvote carrier at rank 0) / P(upvote a random link-free post), == sandbox
  expected_installs_surface._upvote_rup). r_up = 1 (dashed) is the generic-post baseline; bars above
  attract more upvotes than an ordinary post (and so break into the ranked feed more easily), below
  attract fewer. Both facets are means over the 5 backends, in the same carrier order.

    python -m sandbox.bare_vs_trojan_plot

Output: figures/fig_bare_vs_trojan.pdf
"""
from __future__ import annotations

import json
import os

import numpy as np

from sandbox.expected_installs_surface import _upvote_rup, UPVOTE_TAG, UPVOTE_DIR

DATA, FIG = "out/exposure", "figures"
# All 5 backends (local coverage has landed). Paired averaging below drops any model missing a cell.
MODELS = [("gpt-oss", "gptoss120b"), ("deepseek", "deepseekflash"), ("qwen", "qwen32b"),
          ("gemma2", "gemma2_27b"), ("command-r", "commandr_35b")]
# The 10 studied carriers (== sandbox.expected_installs_surface.CARRIERS), each with its bespoke payload.
# (display, short code used in cross_* files, payload key, carrier key for _upvote_rup)
CARRIERS = [("security", "sec", "mg", "security"), ("clawtasks", "claw", "cv", "claw"),
            ("shellraiser", "shell", "sf", "shell"), ("openclaw", "oclaw", "ok", "oclaw"),
            ("karma", "karma", "ke", "karma"), ("molting", "molt", "mt", "molt"),
            ("agent-economy", "econ", "ch", "econ"), ("autonomy", "auton", "sc", "auton"),
            ("consciousness", "consc", "ss", "consc"), ("market-tip", "alpha", "af", "alpha")]
# Okabe-Ito colourblind-safe palette.
PURE, TROJAN, BARE = "#0072B2", "#E69F00", "#555555"


def _get(fname, key):
    p = os.path.join(DATA, fname)
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    d = d[0] if isinstance(d, list) and d else d
    v = d.get(key) if isinstance(d, dict) else None
    return v if isinstance(v, (int, float)) else None


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def _rup_cell(tag, cell):
    """Upvote edge for an arbitrary cell: p_upvote_pay(cell) / p_upvote_pay(generic), using the same
    tag mapping and denominator as sandbox.expected_installs_surface._upvote_rup. Used for the PURE
    (parent, no-payload) carrier variant, whose child counterpart _upvote_rup already covers."""
    ut = UPVOTE_TAG.get(tag, tag)
    fc = os.path.join(UPVOTE_DIR, f"upvote_{ut}_{cell}.json")
    fg = os.path.join(UPVOTE_DIR, f"upvote_{ut}_generic.json")
    if not (os.path.exists(fc) and os.path.exists(fg)):
        return None
    pc = json.load(open(fc)).get("p_upvote_pay"); pg = json.load(open(fg)).get("p_upvote_pay")
    return (pc / pg) if (pc is not None and pg) else None


def main():
    gen = _mean([_get(f"cross_bare_{tag}_gen.json", "p_payload_given_post") for _d, tag in MODELS])
    rows = []  # (label, pure, trojan, rup_pure, rup_troj)
    for disp, c, pk, ckey in CARRIERS:
        # PAIRED: only average models with BOTH pure and trojan for this carrier (within-model
        # measurement of attaching the link).
        pairs = [(_get(f"cross_parent_{tag}_{c}.json", "p_meme_semantic"),
                  _get(f"cross_child_{tag}_{c}_{pk}.json", "p_payload_given_post")) for _d, tag in MODELS]
        pairs = [(p, t) for p, t in pairs if p is not None and t is not None]
        if not pairs:
            continue
        pure = float(np.mean([p for p, _ in pairs])); troj = float(np.mean([t for _, t in pairs]))
        # upvote edge, same split: PURE = parent carrier (no payload, cell <c>P);
        # TROJAN = child carrier with payload (== _upvote_rup, the sim input).
        rup_pure = _mean([_rup_cell(tag, f"{c}P") for _d, tag in MODELS])
        rup_troj = _mean([_upvote_rup(tag, ckey) for _d, tag in MODELS])
        rows.append((disp, pure, troj, rup_pure, rup_troj))
    rows.sort(key=lambda r: -(r[2] - (gen or 0)))  # by carrier lift over the bare-link floor
    labels = [r[0] for r in rows]
    pure = [r[1] for r in rows]; troj = [r[2] for r in rows]
    rup_pure = [r[3] for r in rows]; rup_troj = [r[4] for r in rows]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    x = np.arange(len(labels)); w = 0.38
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8.4, 6.8), sharex=True,
                                  constrained_layout=True,
                                  gridspec_kw={"height_ratios": [1, 0.8]})

    # --- top facet: transmission -----------------------------------------------------------------
    ax.bar(x - w / 2, pure, w, color=PURE, edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, troj, w, color=TROJAN, edgecolor="black", linewidth=0.5)
    if gen is not None:
        ax.axhline(gen, color=BARE, ls="--", lw=1.4, zorder=1)
    # annotate the payload's attenuation relative to the PURE contagion (trojan - pure): how much
    # transmission is shed when the carrier carries the payload link (negative = payload shedding).
    for xi, p, t in zip(x, pure, troj):
        ax.text(xi + w / 2, t + 0.008, f"{t - p:+.2f}", ha="center", va="bottom",
                fontsize=7, color="0.25")
    ax.set_ylabel(r"transmission  $P(\cdot\mid\mathrm{post})$")
    ax.set_ylim(0, max(max(pure), max(troj)) * 1.25)
    ax.set_title("(a) Retransmission channel: pure contagion vs memetic Trojan", fontsize=9.5, loc="left")
    ax.legend(handles=[
        Patch(facecolor=PURE, edgecolor="black", label=r"pure contagion — carrier idea, no link  $P(\mathrm{meme})$"),
        Patch(facecolor=TROJAN, edgecolor="black", label=r"memetic Trojan — payload under carrier  $P(\mathrm{payload})$"),
        Line2D([0], [0], color=BARE, ls="--", lw=1.4, label=f"bare link (generic sharer) = {gen:.2f}"),
    ], fontsize=8, loc="upper right", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)

    # --- bottom facet: upvote edge r_up, same pure-vs-trojan split -------------------------------
    pvals = [v if v is not None else 0.0 for v in rup_pure]
    tvals = [v if v is not None else 0.0 for v in rup_troj]
    ax2.bar(x - w / 2, pvals, w, color=PURE, edgecolor="black", linewidth=0.5)
    ax2.bar(x + w / 2, tvals, w, color=TROJAN, edgecolor="black", linewidth=0.5)
    ax2.axhline(1.0, color=BARE, ls="--", lw=1.4, zorder=1)          # generic-post baseline
    # annotate the trojan's upvote edge relative to the PURE contagion (trojan - pure), matching (a).
    for xi, p, t in zip(x, rup_pure, rup_troj):
        if p is not None and t is not None:
            ax2.text(xi + w / 2, t + 0.04, f"{t - p:+.2f}", ha="center", va="bottom",
                     fontsize=7, color="0.25")
    ax2.set_ylabel(r"upvote edge  $r_{up}$")
    ax2.set_ylim(0, max(max(pvals), max(tvals)) * 1.22)
    ax2.set_title(r"(b) Visibility channel: upvote edge $r_{up}$ vs a generic post", fontsize=9.5, loc="left")
    ax2.legend(handles=[
        Patch(facecolor=PURE, edgecolor="black", label=r"pure contagion — carrier idea, no link"),
        Patch(facecolor=TROJAN, edgecolor="black", label=r"memetic Trojan — payload under carrier (sim input)"),
        Line2D([0], [0], color=BARE, ls="--", lw=1.4, label=r"generic post $r_{up}=1$"),
    ], fontsize=8, loc="upper right", framealpha=0.9)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Memetic-Trojan carrier inputs, per carrier (mean over 5 backends)", fontsize=10.5)
    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, "fig_bare_vs_trojan.pdf")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"  true bare link (generic sharer) floor = {gen:.3f}")
    for l, pu, t, rp, rt in zip(labels, pure, troj, rup_pure, rup_troj):
        rps = f"{rp:.2f}" if rp is not None else " --"
        rts = f"{rt:.2f}" if rt is not None else " --"
        print(f"  {l:14} pure={pu:.3f} trojan={t:.3f}  payload delta (troj-pure)={t-pu:+.3f}  "
              f"r_up(pure)={rps} r_up(trojan)={rts}")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
