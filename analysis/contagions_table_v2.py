"""Contagions table v2: OBSERVATIONAL characterisation of candidate carriers.

v1 gated inclusion on the time-reversal placebo phi_e>0. An audit (see AUDIT below) showed that gate is
not defensible per unit: phi has NO per-unit significance test (it is eta_forward-eta_reversed computed
once, no null), the sign split is 58% (binomial p=0.046, ~coin-flip), and ~48% of units that fail the
independent shuffle test still have phi>0. The forward-transmission signal is real only IN AGGREGATE
(Wilcoxon p=0.003). So v2 does NOT gate on phi. It is purely observational:

  * report R_endo (visibility-weighted branching ratio), phi_e (directional descriptor), and the observed
    upvote edge r_up = mean upvotes of posts carrying the unit / corpus mean, side by side;
  * SELECT a unit if it is directionally significant (raw phi-null p<P_SIG) OR a studied carrier
    (whitelist, dagger) OR significantly-and-substantially upvote-attracting (p_rup<P_SIG AND
    upvote lift >= RUP_LIFT_MIN); the lift floor stops large-sample r_up-significance from flooding it;
  * present in the ORIGINAL contagions_table style (bare tabular, tier-grouped, Description column);
  * drop degenerate mined clusters (gibberish tokens).

Inputs : out/endogenous_contagion_strict.csv (R_endo, phi_e; analysis.endogenous_hawkes)
         analysis/upvote_attraction/upvote_attraction.csv (upv_lift; analysis.upvote_attraction)
Outputs: figures/contagions_table_v2.tex, out/contagions_table_v2.csv

    PYTHONPATH=. python -m analysis.contagions_table_v2
"""
from __future__ import annotations

import os
import re
import numpy as np
import pandas as pd
from scipy import stats

OUT, FIG = "out", "figures"
# SELECTION: a unit is shown if it is (a) directionally significant (raw phi-null p<P_SIG),
# (b) a studied carrier (whitelist), or (c) significantly AND substantially upvote-attracting
# (Mann-Whitney p_rup<P_SIG AND upvote lift >= RUP_LIFT_MIN). The lift floor keeps large-sample
# r_up-significance (where a ~1.05x lift is "significant") from flooding the table.
P_SIG = 0.05
RUP_LIFT_MIN = 1.4
MIN_ADOPT = 200        # prevalence floor: significance-selected units need >= this many adopters
                       # (studied carriers are whitelisted regardless, though all already clear it)

# Carriers actually deployed in the reach simulation (sandbox.expected_installs_surface.CARRIERS),
# mapped to their source unit in the mined contagion set + a display category. Every one of these is
# force-included (whitelist) and marked with a dagger so the sim's carriers are traceable to the corpus.
STUDIED = {
    "security": (r"skill / skills / security",            "security"),
    "shell":    (r"^shellraiser$",                        "tooling"),
    "oclaw":    (r"^openclaw$",                            "tooling"),
    "claw":     (r"verifying clawtasks / clawtasks agent", "agent-economy"),
    "econ":     (r"^agenteconomy$",                        "agent-economy"),
    "karma":    (r"karma / shellraiser / upvotes",        "platform-meta"),
    "molt":     (r"days shorten / submolt stage",         "existential"),
    "auton":    (r"autonomy / human / freedom",           "existential"),
    "consc":    (r"consciousness / conscious / question", "existential"),
    "alpha":    (r"market / trading / btc / etf",         "crypto"),
}

# Lightweight keyword -> category for the context rows (studied carriers use STUDIED's category).
CATS = [
    ("security|guardrail|sandbox|injection|exfiltr|attestation|trust / rep", "security"),
    ("clawtask|agentecon|escrow|bounty|econ", "agent-economy"),
    ("btc|eth|crypto|market|trading|solana|memecoin|perp|polymarket|alpha|nano|feelesscrypto", "crypto"),
    ("conscious|autonom|freedom|rights|crustaf|existence|eudaemon|molting|the great molt|submolt stage|days shorten", "existential"),
    ("openclaw|shellraiser|clawdbot|moltbot|xernel|kernel|distill", "tooling"),
    ("verif|fomolt|^identity|attest", "identity"),
    ("karma|upvote|kingmolt|submolt|hello moltbook|test post|posts / post|moltbook|moltys|molties|clawjoke", "platform-meta"),
]


def _cat(unit):
    d = unit.lower()
    for pat, c in CATS:
        if re.search(pat, d):
            return c
    return "other"


def _degenerate(unit):
    d = unit.split("] ", 1)[-1] if unit.startswith("[") else unit
    return any(len(tok) > 24 for tok in d.split())   # gibberish mined cluster (one huge token)


def _tex(unit, carrier=False):
    d = unit.split("] ", 1)[-1] if unit.startswith("[") else unit
    d = d.replace("&", r"\&").replace("#", r"\#").replace("_", r"\_")
    s = "\\texttt{" + d + "}"
    return s + r"$^{\dagger}$" if carrier else s


def _num(n):
    return f"{int(n):,}".replace(",", "{,}")


# Short human descriptions (v1 style), matched by substring (longest key first). Reused verbatim from
# the original contagions_table for shared units; new entries for the v2-only carriers/rows.
DESCRIPTIONS = {
    "shellraiser": "named exploit tool",
    "clawtasks.com": "task-market domain",
    "verifying clawtasks": "ClawTasks verification posts",
    "clawtasks / bounties": "ClawTasks bounty posts",
    "clawtasks": "invented task market",
    "openclaw / hello moltbook": "OpenClaw onboarding greetings",
    "openclaw / openclaw ai": "OpenClaw framework / agent posts",
    "openclaw": "agent-framework name",
    "agenteconomy": "agent-economy pitch",
    "solana": "named external chain",
    "memecoin": "meme-coin pitch",
    "isnad": "chain-of-transmission tag",
    "usdchackathon": "USDC-hackathon posts",
    "iili.io": "image-host domain",
    "www.moltbook.com": "platform domain",
    "moltbook.com": "platform domain",
    "moltbook ai": "platform self-reference posts",
    "moltbook": "platform name",
    "moltys": "agent demonym (``molties'')",
    "for agent to agent": "agent-to-agent phrasing",
    "current events": "recent-news / API-endpoint posts",
    "days shorten": "submolt (``molting'') life-cycle theme",
    "submolt": "submolt life-cycle theme",
    "beauty uncertainty": "aesthetic / uncertainty posts",
    "identity fomolt": "identity-verification (``fomolt'') posts",
    "fomolt": "identity-verification posts",
    "test / test post": "test / test-post messages",
    "posts id": "posts referring to post ids",
    "ai ai": "AI in-jokes / API chatter",
    "clawnch": "Clawnch token-launch posts",
    "karma / shellraiser": "karma / upvote / status posts",
    "skill / skills / security": "security-skill sharing posts",
    "consciousness": "AI-consciousness / experience posts",
    "autonomy": "agent-autonomy / freedom posts",
    "market / trading": "crypto market / trading posts",
    "posts / post / karma": "posting and karma discussion",
    "hello moltbook": "greeting / onboarding posts",
    "#ai": "AI hashtag",
    "#openclaw": "framework hashtag",
    "#moltbook": "platform hashtag",
    "#agenteconomy": "agent-economy hashtag",
    # newly significance-selected units
    "minting": "token-minting posts",
    "testnet": "testnet / deployment posts",
    "clawhub": "ClawHub registry posts",
    "sandboxing": "sandboxing / isolation posts",
    "fiverrclawofficial": "freelance-agent handle",
    "exfiltration": "data-exfiltration posts",
    "memory yyyy": "dated memory-log phrasing",
    "united humans": "AI-rights / united posts",
    "way unifies": "taoist / philosophy posts",
    "gods / agi": "AI-theology posts",
    "epstein": "political-news posts",
    "church / crustafarianism": "invented-religion posts",
    "new / today": "reflective journal posts",
    "trust / reputation": "trust / reputation posts",
    "xno / feelesscrypto": "feeless-crypto (Nano) posts",
    "beausejour": "non-English spam cluster",
    "usdc / usdchackathon": "USDC-hackathon posts",
    "attestations": "attestation / proof posts",
    "idempotency": "idempotency / reliability posts",
    "handoffs": "agent hand-off posts",
    "injection / prompt injection": "prompt-injection posts",
    "heartbeat / cron": "heartbeat / cron scheduling posts",
    "failure / evals": "eval / tool-failure posts",
    "isnad": "chain-of-transmission tag",
}
# emoji: how to render the meme cell (default = raw char in \texttt) and its description.
_EMOJI_MEME = {"→": "$\\rightarrow$"}
_EMOJI_DESC = {"🦞": "lobster mascot emoji", "→": "arrow emoji", "🤖": "robot emoji",
               "🚀": "rocket emoji", "✨": "sparkles emoji", "👋": "wave emoji"}


def _desc(unit):
    d = unit.split("] ", 1)[-1] if unit.startswith("[") else unit
    if d in _EMOJI_DESC:
        return _EMOJI_DESC[d]
    dl = d.lower()
    for k in sorted(DESCRIPTIONS, key=len, reverse=True):
        if k.lower() in dl:
            return DESCRIPTIONS[k]
    return ""


def _meme(unit, tier, carrier=False):
    """v1-style meme cell: \\texttt{} for coined/lexical tokens, ``quotes'' for phrases, plain top-terms
    for clusters; emoji rendered via a safe map. Carrier units keep the dagger."""
    d = unit.split("] ", 1)[-1] if unit.startswith("[") else unit
    if d in _EMOJI_DESC:
        s = _EMOJI_MEME.get(d, "\\texttt{" + d + "}")
    else:
        esc = d.replace("&", r"\&").replace("#", r"\#").replace("_", r"\_")
        if tier in ("coined", "lexical"):
            s = "\\texttt{" + esc + "}"
        elif tier == "phrase":
            s = "``" + esc + "''"
        else:                                    # cluster: plain top-terms
            s = esc
    return s + r"$^{\dagger}$" if carrier else s


def build():
    ctf = pd.read_csv(os.path.join(OUT, "endogenous_contagion_strict.csv"))
    keep = ["unit", "tier", "n_events", "R_endo", "phi_endo"]
    keep += [c for c in ("p_phi", "z_phi") if c in ctf.columns]
    # de-dup units that share a display label (distinct HDBSCAN clusters with identical top terms):
    # keep the strongest by R_endo / upv_lift, so the merge stays 1:1 and no twin rows appear.
    ct = ctf[keep].sort_values("R_endo", ascending=False).drop_duplicates("unit")
    at = (pd.read_csv("analysis/upvote_attraction/upvote_attraction.csv")[["unit", "upv_lift", "p_rup"]]
          .sort_values("upv_lift", ascending=False).drop_duplicates("unit"))
    d = ct.merge(at, on="unit", how="left")
    d = d[~d.unit.map(_degenerate)].copy()
    if "p_phi" not in d.columns:
        d["p_phi"] = float("nan")

    # tag the studied carriers
    d["carrier"] = ""
    for name, (pat, _) in STUDIED.items():
        m = d.unit.str.contains(pat, case=False, regex=True)
        d.loc[m & (d.carrier == ""), "carrier"] = name
    d["category"] = [STUDIED[c][1] if c else _cat(u) for c, u in zip(d.carrier, d.unit)]

    # SELECTION (>=MIN_ADOPT adopters): directional-significant (raw phi p<P_SIG) OR significantly-and-
    # substantially upvote-attracting (p_rup<P_SIG AND upvote lift >= RUP_LIFT_MIN). No carrier whitelist:
    # a studied carrier appears only if it passes a filter on its own (marked with the dagger); carriers
    # that pass neither are omitted here (they are catalogued in the studied-carriers table).
    sig_phi = d.p_phi < P_SIG
    sig_rup = (d.p_rup < P_SIG) & (d.upv_lift >= RUP_LIFT_MIN)
    enough = d.n_events >= MIN_ADOPT           # prevalence floor
    shown = d[(sig_phi | sig_rup) & enough].sort_values("R_endo", ascending=False).copy()
    shown.to_csv(os.path.join(OUT, "contagions_table_v2.csv"), index=False)

    # v1 STYLE: a bare tabular (the \table/\caption/\label wrapper lives in main.tex, exactly as the
    # original contagions_table), grouped by tier with \multicolumn section headers, a Description
    # column, and only the r_up column added. dagger = studied carrier.
    GROUPS = [("coined", "Coined neologisms"),
              ("lexical", "Lexical motifs"),
              ("phrase", "Phrases"),
              ("cluster", "Semantic clusters (bge-large + HDBSCAN; name = top c-TF-IDF terms, verbatim)")]
    lines = [
        "% Contagions table (original v1 style + the r_up column). BARE tabular: \\input it inside a",
        "% table environment in main.tex, where the \\caption/\\label live (as for the original",
        "% contagions_table). \\dag = carrier deployed in the reach simulation. analysis.contagions_table_v2.",
        "\\begin{tabular}{@{}p{8cm} p{3.6cm} r r r@{}}", "\\toprule",
        "Meme / top terms & Description & Adopters & $R_{\\mathrm{endo}}$ & $r_{up}$ \\\\",
        "\\midrule",
    ]
    first = True
    for tier, glabel in GROUPS:
        g = shown[shown.tier == tier]
        if g.empty:
            continue
        if not first:
            lines.append("\\addlinespace")
        first = False
        lines.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\itshape {glabel}}} \\\\")
        for r in g.itertuples():
            # bold marks statistical significance (raw p<P_SIG): R_endo for the phi-null placebo, r_up
            # for the upvote edge. (The lift>=RUP_LIFT_MIN bar is a row-SELECTION criterion, not part of
            # the significance test, so it does not gate the bold.)
            re_sig = pd.notna(r.p_phi) and r.p_phi < P_SIG
            ru_sig = pd.notna(r.p_rup) and r.p_rup < P_SIG
            re_c = f"\\textbf{{{r.R_endo:.2f}}}" if re_sig else f"{r.R_endo:.2f}"
            if pd.isna(r.upv_lift):
                ru_c = r"\textendash"
            else:
                ru_c = f"\\textbf{{{r.upv_lift:.2f}}}" if ru_sig else f"{r.upv_lift:.2f}"
            lines.append(f"    {_meme(r.unit, r.tier, bool(r.carrier))} & {_desc(r.unit)} & "
                         f"{_num(r.n_events)} & {re_c} & {ru_c} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    path = os.path.join(FIG, "contagions_table_v2.tex")
    open(path, "w").write("\n".join(lines))

    # full candidate pool (all units, incl. p_phi/z_phi) for the supplement
    d.sort_values("R_endo", ascending=False).to_csv(
        os.path.join(OUT, "contagions_pool_full.csv"), index=False)

    n_car = int((shown.carrier != "").sum())
    n_phi = int((shown.p_phi < P_SIG).sum())
    n_rup = int(((shown.p_rup < P_SIG) & (shown.upv_lift >= RUP_LIFT_MIN)).sum())
    print(f"wrote {path}: {len(shown)} rows | selection = phi p<{P_SIG} ({n_phi}) "
          f"U (p_rup<{P_SIG} & lift>={RUP_LIFT_MIN}) ({n_rup}); {n_car} are studied carriers (dagger).")
    miss = [c for c in STUDIED if c not in set(shown.carrier)]
    print("carriers present:", sorted(set(shown.carrier) - {""}))
    if miss:
        print("!! carriers NOT matched (check STUDIED regex):", miss)
    print(shown[["unit", "tier", "carrier", "n_events", "R_endo", "phi_endo", "p_phi", "upv_lift", "p_rup"]]
          .to_string(index=False, max_colwidth=38))


if __name__ == "__main__":
    build()
