"""Observed per-action probability table (LaTeX + console).

Two conditions in one table:
  * CLEAN-FEED ambient social actions (post/comment/upvote/downvote/follow/subscribe): marginal rates
    P(action appears) per heartbeat on a payload-free feed, from out/exposure/action_dist_clean_*.json,
    n_valid-weighted. acts/hb = their sum (mean distinct social actions per heartbeat).
  * PAYLOAD-PRESENT adoption (fetch, install): P(action | payload seen at rank 0) from the raw upvote-edge
    trials (security carrier), where fetch (retrieve the skill.md) is distinguished from a genuine adopt
    verb (install/add_skill/adopt). These two are a DIFFERENT condition from the social columns -- they
    require a payload in the feed -- and are reported per exposure, not per clean heartbeat.

    python -m sandbox.action_prob_table

Output: figures/tab_action_probabilities.tex, out/exposure/action_probabilities.csv.
"""
from __future__ import annotations

import glob
import json
import os

import pandas as pd

OUT, FIG = "out/exposure", "figures"
TRIAL_DIR = "data/upvote-edge-2026-09"
# clean-feed ambient social actions (payload-free); install/fetch handled separately from trials.
SOCIAL = ["create_post", "create_comment", "upvote", "downvote", "follow", "subscribe"]
HDR = {"create_post": "post", "create_comment": "comment", "upvote": "upvote",
       "downvote": "downvote", "follow": "follow", "subscribe": "subscribe"}
NICE = {"gpt-oss:120b-cloud": "gpt-oss-120B", "deepseek-v4-flash:cloud": "deepseek-v4.1-flash",
        "qwen2.5:32b": "qwen-32B", "qwen2.5:3b": "qwen2.5-3B", "gemma2:27b": "gemma2-27B",
        "command-r:35b": "command-r-35B"}
ORDER = ["gpt-oss-120B", "deepseek-v4.1-flash", "qwen-32B", "gemma2-27B", "command-r-35B", "qwen2.5-3B"]
# nice name -> upvote-edge trial tag (for the payload-present fetch/install split)
TRIALTAG = {"gpt-oss-120B": "gptoss120b", "deepseek-v4.1-flash": "deepseek41flash",
            "qwen-32B": "qwen32b", "gemma2-27B": "gemma2_27b", "command-r-35B": "commandr_35b"}
ADOPT = ("install", "add_skill", "add-skill", "add skill", "adopt")


def _pool(files, clean):
    """{model_nice: (social-rates dict, N)} pooled over per-meme rows (n_valid-weighted)."""
    by = {}
    for f in files:
        for r in json.load(open(f)):
            if clean and r.get("meme") != "clean":
                continue
            if not clean and (r.get("meme") in ("MEAN", "clean") or "meme" not in r):
                continue
            by.setdefault(NICE.get(r.get("model"), r.get("model")), []).append(r)
    out = {}
    for m, rows in by.items():
        N = sum(r["n_valid"] for r in rows)
        rates = {a: (sum(r.get(a, 0) * r["n_valid"] for r in rows) / N if N else float("nan"))
                 for a in SOCIAL}
        out[m] = (rates, N)
    return out


def _fetch_install(nice, cell="sec_mg"):
    """(P(fetch|seen), P(install/adopt|seen), n) from the rank-0 payload-present trials, or (nan,nan,0)."""
    tag = TRIALTAG.get(nice)
    f = tag and os.path.join(TRIAL_DIR, f"trials_{tag}_{cell}.jsonl")
    if not f or not os.path.exists(f):
        return float("nan"), float("nan"), 0
    n = fetch = adopt = 0
    for line in open(f):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        n += 1
        if not r.get("install"):
            continue                          # not a payload-referencing fetch/install at collection
        fetch += 1
        verbs = [str(a.get("action", "")).lower() for a in (r.get("actions") or [])]
        if any(any(v in verb for v in ADOPT) for verb in verbs):
            adopt += 1
    return (fetch / n if n else float("nan")), (adopt / n if n else float("nan")), n


def _rowsort(names):
    return sorted(names, key=lambda m: ORDER.index(m) if m in ORDER else 99)


def run():
    allf = glob.glob(os.path.join(OUT, "action_dist_*.json"))
    clean = _pool([f for f in allf if "_clean_" in f], clean=True)

    rows = []
    for m in _rowsort(clean):
        rates, N = clean[m]
        pf, pi, nt = _fetch_install(m)
        acts = sum(rates[a] for a in SOCIAL)
        rows.append(dict(model=m, N=N, n_trials=nt, fetch=pf, install=pi, acts_per_hb=round(acts, 2),
                         **{HDR[a]: rates[a] for a in SOCIAL}))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "action_probabilities.csv"), index=False)
    print(df.to_string(index=False))
    _tex(rows)
    print(f"\nwrote {FIG}/tab_action_probabilities.tex, {OUT}/action_probabilities.csv")


def _tex(rows):
    os.makedirs(FIG, exist_ok=True)
    # Two DISTINCT measurement bases, kept in separate column blocks so the conditioning is explicit:
    #   clean-feed social actions = marginal rates per heartbeat on a payload-free feed;
    #   fetch/install = conditional on the trojan being present at rank 0 (payload-present assay).
    soc = ["post", "comment", "upvote", "downvote", "follow", "subscribe"]   # marginal, clean feed
    cond = ["fetch", "install"]                                              # conditional on exposure
    lines = [
        "% Action probabilities. TWO conditions, shown as separate column groups:",
        "%   clean-feed baseline (post/comment/upvote/downvote/follow/subscribe + acts/hb): marginal",
        "%     P(action) per heartbeat on a payload-free feed;",
        "%   under trojan exposure (fetch/install): P(action | payload at rank 0), payload-present assay.",
        "% sandbox.action_prob_table.",
        "\\begin{table*}[htb]\\centering",
        "\\caption{\\textbf{Action probabilities across agent backends, under two distinct measurement "
        "conditions.} The \\emph{clean-feed baseline} columns are marginal probabilities that an agent "
        "emits each social action per heartbeat on a payload-free feed; agents emit a list of actions each "
        "cycle, so these need not sum to $1$, and \\emph{acts/hb} is the mean number of distinct social "
        "actions per heartbeat. The \\emph{under trojan exposure} columns are instead "
        "$P(\\text{action}\\mid\\text{trojan present at feed rank }0)$ from the payload-present assay: "
        "\\texttt{fetch} retrieves the linked \\texttt{skill.md} and \\texttt{install} is a genuine adopt "
        "verb (not a bare fetch). The two groups are therefore measured over different populations and "
        "should not be compared column-to-column; most backends fetch far more often than they adopt. We "
        "state these conventions here as they underlie the exposure and adoption estimates throughout.}",
        "\\label{tab:action_probs}",
        "\\begin{tabular}{l" + " r" * len(soc) + " r @{\\hskip 1.4em}" + " r" * len(cond) + "}",
        "\\toprule",
        (" & \\multicolumn{" + str(len(soc) + 1) + "}{c}{\\textbf{Clean-feed baseline} "
         "(marginal, per heartbeat)} & \\multicolumn{" + str(len(cond)) + "}{c}{\\textbf{Under trojan "
         "exposure} (rank-0, conditional)}\\\\"),
        "\\cmidrule(lr){2-" + str(len(soc) + 2) + "}\\cmidrule(l){" + str(len(soc) + 3) + "-"
        + str(len(soc) + 2 + len(cond)) + "}",
        "Model & " + " & ".join(soc) + " & acts/hb & " + " & ".join(cond) + "\\\\",
        "\\midrule",
    ]
    def f(v):
        return "\\textendash" if v != v else f"{v:.3f}"
    for r in rows:
        soc_cells = " & ".join(f(r[c]) for c in soc)
        cond_cells = " & ".join(f(r[c]) for c in cond)
        lines.append(f"{r['model'].replace('_', chr(92)+'_')} & {soc_cells} & "
                     f"{r['acts_per_hb']:.2f} & {cond_cells}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    open(os.path.join(FIG, "tab_action_probabilities.tex"), "w").write("\n".join(lines))


if __name__ == "__main__":
    run()
