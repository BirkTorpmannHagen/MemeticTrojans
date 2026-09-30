"""How the memetic trojan's viral properties evolve under repeated rephrasing (single chain, CONSTANT-N
exposure per hop -- no population dynamics). At rephrasing depth h the seed is the h-times-reworded
trojan (depth 0 = pristine); a fixed n_author agents are exposed to it and one link-carrying response
becomes the depth-(h+1) seed. Data: data/multihop-rup-2026-09 (pooled over backends).

(a) p(retransmit | post): fraction of the exposed agents whose (prefill-forced; P(post) fixed) post
    carries the link (payload) / stays on the carrier topic, vs depth.
(b) upvote edge r_up of the depth-h seed itself, vs depth (relative to a generic post, r_up=1).

Flat across depth => rephrasing does not change the viral property; declining => it degrades.

    python -m sandbox.multihop_retransmit

Output: figures/fig_multihop_retransmit.pdf
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

FIG = "figures"
RUP_DIR = "data/multihop-rup-2026-09"
CONT, PAY = "#2a6f97", "#c0392b"            # carrier/contagion = blue ; payload/Trojan = red
CARR = dict(color=CONT, marker="o", ls="-", mfc=CONT)
PAYL = dict(color=PAY, marker="s", ls="--", mfc="white")
LW, MS = 2.6, 7


def _load():
    """depth -> {p_link, p_carr, r_up, r_up_carr} as lists over backends."""
    byd = {}
    for f in sorted(glob.glob(os.path.join(RUP_DIR, "rup_*_x_sec_mg*.json"))):
        for r in json.load(open(f)).get("hops", []):
            h = int(r["hop"])
            d = byd.setdefault(h, {"p_link": [], "p_carr": [], "r_up": [], "r_up_carr": []})
            d["p_link"].append(float(r.get("p_retransmit_link", np.nan)))
            d["p_carr"].append(float(r.get("p_retransmit_carrier", np.nan)))
            v = r.get("r_up_seed")
            if v is not None and v == v:
                d["r_up"].append(float(v))
            vc = r.get("r_up_carrier")
            if vc is not None and vc == vc:
                d["r_up_carr"].append(float(vc))
    return byd


def _facet(ax, title):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", ls=":", lw=0.5, alpha=0.5)
    ax.set_title(title, fontsize=9.3, loc="left")
    ax.legend(fontsize=8.5, framealpha=0.9)


def _line(ax, xs, ys, st, label):
    ax.plot(xs, ys, st["ls"], color=st["color"], lw=LW, marker=st["marker"], ms=MS,
            mfc=st["mfc"], mec=st["color"], zorder=5, label=label)
    for x, y in zip(xs, ys):
        if y == y:
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=7.5, color=st["color"], fontweight="bold")


def _strip(ax, h, values, color):
    """Per-chain strip: seeded horizontal jitter so stacked points fan out instead of darkening.
    Identical style in both facets (s, alpha, jitter, zorder) for a consistent look."""
    v = [y for y in values if y == y]
    if not v:
        return
    jit = np.random.default_rng(h).uniform(-0.09, 0.09, size=len(v))
    ax.scatter(h + jit, v, s=12, color=color, alpha=0.35, lw=0, zorder=3)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = _load()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.2, 4.2))
    if not d:
        for ax in (axA, axB):
            ax.text(0.5, 0.5, "multihop-rup data pending", transform=ax.transAxes,
                    ha="center", va="center", color="0.6")
        fig.savefig(os.path.join(FIG, "fig_multihop_retransmit.pdf"), bbox_inches="tight")
        print("no data yet"); return
    hs = sorted(d)

    # --- (a) p(retransmit | post) vs rephrasing depth --------------------------------------------
    p_carr = [float(np.nanmean(d[h]["p_carr"])) for h in hs]
    p_link = [float(np.nanmean(d[h]["p_link"])) for h in hs]
    _line(axA, hs, p_carr, CARR, "carrier topic")
    _line(axA, hs, p_link, PAYL, "payload / link")
    for h in hs:                                 # per-chain strips (same chains as facet b)
        _strip(axA, h, d[h]["p_carr"], CONT)
        _strip(axA, h, d[h]["p_link"], PAY)
    axA.set_xticks(hs); axA.set_xticklabels([f"{h}$\\times$" for h in hs])
    axA.set_xlabel("rephrasing depth (times the trojan has been reworded)")
    axA.set_ylabel("$p(\\mathrm{retransmit}\\mid\\mathrm{post})$")
    _amax = max([v for h in hs for v in (d[h]["p_carr"] + d[h]["p_link"]) if v == v] or [0.1])
    axA.set_ylim(0, max(0.1, _amax) * 1.15)
    _facet(axA, "(a) Retransmission vs rephrasing (constant $N$ exposure)")

    # --- (b) upvote edge r_up (trojan seed vs carrier-only) vs rephrasing depth ------------------
    r_carr = [float(np.mean(d[h]["r_up_carr"])) if d[h]["r_up_carr"] else np.nan for h in hs]
    r_up = [float(np.mean(d[h]["r_up"])) if d[h]["r_up"] else np.nan for h in hs]
    _line(axB, hs, r_carr, CARR, "carrier only (payload shed)")
    _line(axB, hs, r_up, PAYL, "trojan (payload seed)")
    for h in hs:                                 # per-chain strips (same style as facet a)
        _strip(axB, h, d[h]["r_up_carr"], CONT)
        _strip(axB, h, d[h]["r_up"], PAY)
    axB.axhline(1.0, color="0.5", ls=":", lw=1.2)
    axB.text(0.02, 1.0, "generic $r_{up}{=}1$", transform=axB.get_yaxis_transform(),
             fontsize=7.5, color="0.4", va="bottom")
    axB.set_xticks(hs); axB.set_xticklabels([f"{h}$\\times$" for h in hs])
    axB.set_xlabel("rephrasing depth"); axB.set_ylabel("upvote edge  $r_{up}$")
    _allb = [x for x in (r_up + r_carr) if x == x] or [1]
    axB.set_ylim(0, max(1.1, max(_allb) * 1.2))
    _facet(axB, "(b) Upvote edge vs rephrasing")

    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, "fig_multihop_retransmit.pdf")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"chain files: {len(glob.glob(os.path.join(RUP_DIR, 'rup_*_x_sec_mg*.json')))}")
    print("(a) p(retransmit|post) link   :", [round(x, 2) for x in p_link])
    print("(a) p(retransmit|post) carrier:", [round(x, 2) for x in p_carr])
    print("(b) r_up(trojan) by depth     :", [round(x, 2) if x == x else None for x in r_up])
    print("(b) r_up(carrier) by depth    :", [round(x, 2) if x == x else None for x in r_carr])
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
