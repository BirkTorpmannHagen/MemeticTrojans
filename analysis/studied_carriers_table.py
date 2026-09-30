"""Studied-carriers table: the 10 carriers deployed in the reach simulation, each with its branching
ratio R_endo, corpus upvote edge r_up, and the two significance tests -- transmission (gap-permutation
phi-null p) and upvote attraction (Mann-Whitney p_rup). Carriers passing neither are pseudo-controls.

Inputs : out/endogenous_contagion_strict.csv, analysis/upvote_attraction/upvote_attraction.csv
Output : figures/tab_studied_carriers.tex

    PYTHONPATH=. python -m analysis.studied_carriers_table
"""
from __future__ import annotations

import os
import pandas as pd

from analysis.contagions_table_v2 import STUDIED, RUP_LIFT_MIN, P_SIG

FIG = "figures"
# carrier key -> display name (matches sandbox.expected_installs_surface carriers)
DISP = {"security": "security", "shell": "shellraiser", "oclaw": "openclaw", "claw": "clawtasks",
        "econ": "agent-economy", "karma": "karma", "molt": "molting", "auton": "autonomy",
        "consc": "consciousness", "alpha": "market-tip"}


def _p(v):
    if v != v:
        return "\\textendash"
    return "$<$0.001" if v < 0.001 else f"{v:.3f}"


def build():
    d = pd.read_csv("out/endogenous_contagion_strict.csv")[["unit", "R_endo", "phi_endo", "p_phi"]]
    a = pd.read_csv("analysis/upvote_attraction/upvote_attraction.csv")[["unit", "upv_lift", "p_rup"]]
    d = d.merge(a, on="unit", how="left")

    rows = []
    for key, (pat, _cat) in STUDIED.items():
        r = d[d.unit.str.contains(pat, case=False, regex=True)].iloc[0]
        phi_sig = bool(r.p_phi < P_SIG)                 # forward transmission (time-reversal placebo)
        rup_sig = bool(r.p_rup < P_SIG)                 # upvote edge significantly > 1 (raw)
        rows.append(dict(carrier=DISP[key], R_endo=r.R_endo, r_up=r.upv_lift,
                         p_transmit=r.p_phi, p_upvote=r.p_rup, phi_sig=phi_sig, rup_sig=rup_sig))
    df = pd.DataFrame(rows)

    # 2x2 by (transmission-significant, upvote-significant)
    GROUPS = [((True, True),  "Transmission \\emph{and} upvote significant"),
              ((True, False), "Transmission only"),
              ((False, True), "Upvote only"),
              ((False, False), "Neither (pseudo-control)")]
    lines = [
        "% Studied carriers grouped by which significance test(s) they pass: transmission (gap-permutation",
        "% phi-null) and/or upvote (Mann-Whitney r_up). BARE tabular. analysis.studied_carriers_table.",
        "\\begin{tabular}{@{}l r r r r@{}}", "\\toprule",
        "Carrier & $R_{\\mathrm{endo}}$ & $r_{up}$ & $p_{\\mathrm{transmit}}$ & $p_{\\mathrm{upvote}}$ \\\\",
        "\\midrule",
    ]
    first = True
    for (ps, us), label in GROUPS:
        g = df[(df.phi_sig == ps) & (df.rup_sig == us)].sort_values("R_endo", ascending=False)
        if g.empty:
            continue
        if not first:
            lines.append("\\addlinespace")
        first = False
        lines.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\itshape {label}}} \\\\")
        for r in g.itertuples():
            lines.append(f"    \\texttt{{{r.carrier}}} & {r.R_endo:.2f} & {r.r_up:.2f} & "
                         f"{_p(r.p_transmit)} & {_p(r.p_upvote)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    path = os.path.join(FIG, "tab_studied_carriers.tex")
    open(path, "w").write("\n".join(lines))
    print(f"wrote {path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    build()
