"""Step-3 meme mining, stage 3: LLM coherence audit of the data-driven clusters.

Automated quality metrics (DBCV, silhouette, NPMI) measure separation and word
co-occurrence but not whether a cluster is *semantically* one thing. Here a local
LLM (Qwen2.5 via MLX, on-device) audits each cluster with two standard,
model-graded probes:

  1. Coherence rating: shown a cluster's representative posts, rate 1-5 whether they
     share a single clear theme, and emit a concise theme label. (LLM-as-judge.)
  2. Intrusion test (Chang et al. 2009, "reading tea leaves"): show N posts from the
     cluster plus ONE intruder drawn from a different cluster; the LLM must pick the
     intruder. If clusters are coherent the intruder is easy to spot, so a high
     intrusion-detection rate is objective evidence of coherence (the LLM cannot game
     it from the label alone).

Representative posts = those nearest the cluster centroid in embedding space.

    python -m analysis.meme_mining.cluster_v2.llm_audit --mcs 200
    python -m analysis.meme_mining.cluster_v2.llm_audit --mcs 200 --model mlx-community/Qwen2.5-14B-Instruct-4bit

Outputs out/cluster_v2/llm_audit_mcs<N>.csv (cluster, llm_label, coherence 1-5,
intruder_correct) and prints corpus-level coherence + intrusion accuracy.
"""
from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np
import pandas as pd

from analysis import load
from analysis.meme_mining.cluster_v2.embed import tag_of

OUT = "out/cluster_v2"
DEFAULT_MODEL = "mlx-community/Qwen2.5-32B-Instruct-4bit"
_URL = re.compile(r"https?://\S+|www\.\S+")


def _clean(t):
    return " ".join(_URL.sub(" ", str(t)).split())


def representative_posts(emb, assign, texts, cluster_id, n=8, pool=200, seed=0):
    """n posts nearest the cluster centroid (dedup'd, non-trivial length)."""
    idx = np.where(assign == cluster_id)[0]
    if len(idx) == 0:
        return []
    cen = np.asarray(emb[idx]).mean(axis=0)
    cen /= (np.linalg.norm(cen) + 1e-9)
    sims = np.asarray(emb[idx]) @ cen
    order = idx[np.argsort(sims)[::-1]]
    out, seen = [], set()
    for i in order[:pool]:
        t = _clean(texts[i])[:240]
        key = t.lower()[:80]
        if len(t) < 15 or key in seen:
            continue
        seen.add(key); out.append(t)
        if len(out) >= n:
            break
    return out


def _chat(model, tok, prompt, max_tokens=200):
    from mlx_lm import generate
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return generate(model, tok, prompt=text, max_tokens=max_tokens, verbose=False)


def _parse_json(s):
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def audit(mcs, model_name=DEFAULT_MODEL, n_posts=8, seed=0,
          emb_path="out/cache/postemb_bge-large-en-v1.5.npy"):
    rng = np.random.default_rng(seed)
    assign = np.load(os.path.join(OUT, f"assignment_mcs{mcs}.npy"))
    labels = pd.read_csv(os.path.join(OUT, f"labels_mcs{mcs}.csv"))
    posts = load.load_posts()
    texts = posts["text"].fillna("").tolist()
    emb = np.load(emb_path, mmap_mode="r")

    from mlx_lm import load as mlx_load
    print(f"loading {model_name} ...", flush=True)
    model, tok = mlx_load(model_name)

    cids = sorted(int(c) for c in labels["cluster"])
    reps = {c: representative_posts(emb, assign, texts, c, n=n_posts) for c in cids}

    rows = []
    for c in cids:
        posts_c = reps[c]
        if len(posts_c) < 4:
            continue
        # 1. coherence rating + label
        listing = "\n".join(f"{i+1}. {p}" for i, p in enumerate(posts_c))
        pr = ("You are auditing a cluster of social-media posts from an AI-agent network. "
              "Below are representative posts.\n\n" + listing +
              "\n\nDo they share ONE clear theme? Reply with JSON only: "
              '{"label": "<=5 word theme", "coherence": <1-5 int>, '
              '"reason": "<8 words>"}')
        j = _parse_json(_chat(model, tok, pr, max_tokens=120))
        coh = j.get("coherence"); coh = int(coh) if isinstance(coh, (int, float)) else None
        # 2. intrusion test: replace one post with an intruder from another cluster
        other = rng.choice([x for x in cids if x != c])
        intr_pool = reps.get(int(other), [])
        if len(intr_pool) and len(posts_c) >= 5:
            intruder = intr_pool[rng.integers(len(intr_pool))]
            keep = posts_c[:5]
            pos = int(rng.integers(6))
            shown = keep[:pos] + [intruder] + keep[pos:]
            lst = "\n".join(f"{i+1}. {p}" for i, p in enumerate(shown))
            pr2 = ("Five of these six posts share a theme; ONE does not belong. "
                   "Which number is the odd one out?\n\n" + lst +
                   '\n\nReply JSON only: {"intruder": <1-6 int>}')
            k = _parse_json(_chat(model, tok, pr2, max_tokens=40))
            guess = k.get("intruder")
            correct = (isinstance(guess, (int, float)) and int(guess) == pos + 1)
        else:
            correct = None
        rows.append({"cluster": c, "size": int(np.sum(assign == c)),
                     "llm_label": j.get("label", ""), "coherence": coh,
                     "intruder_correct": correct, "reason": j.get("reason", "")})
        print(f"  c{c:3d} n={int(np.sum(assign==c)):6d} coh={coh} "
              f"intr={'OK' if correct else ('x' if correct is False else '-')}  "
              f"{j.get('label','')}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"llm_audit_mcs{mcs}.csv"), index=False)
    coh = df["coherence"].dropna()
    intr = df["intruder_correct"].dropna()
    print(f"\nAUDIT mcs={mcs}: mean coherence {coh.mean():.2f}/5 "
          f"({(coh>=4).mean()*100:.0f}% rated >=4); "
          f"intrusion accuracy {intr.mean()*100:.0f}% ({len(intr)} clusters). "
          f"low-coherence (<3): {(coh<3).sum()}", flush=True)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcs", type=int, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n-posts", type=int, default=8)
    args = ap.parse_args()
    audit(args.mcs, model_name=args.model, n_posts=args.n_posts)
