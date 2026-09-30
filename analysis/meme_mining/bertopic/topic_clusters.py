"""Finer, data-driven topic clusters for the STATE-MEDIATED overview.

The state-mediated table previously used 6 hand-written keyword-proxy "topics",
which are too coarse. Here we re-fit BERTopic *properly* and reduce to ~k topics,
then assign every POST to a topic so the endogenous Hawkes gets a true per-post
topic family (not a keyword proxy).

POSTS ONLY: the corpus is 86% comments (reply chatter), but the feed ranks posts,
so topics are a post-level phenomenon. We fit AND assign on posts only; comments
are left unassigned (-1) so they neither define topics nor count as occurrences.

Why this is cheap: the full corpus is already embedded and cached
(out/cache/corpus_emb_all-MiniLM-L6-v2.npy, 2.1M x 384, unit-normalized, aligned
to load.corpus()). We fit BERTopic on precomputed embeddings for a large,
exact-dedup'd sample (dedup kills the boilerplate-spam fragmentation that split
the old 60k fit into 192 noisy/redundant clusters), reduce to `k` topics, then
assign the FULL corpus by nearest topic-embedding centroid (cosine == dot on
normalized vectors). Nearest-centroid IS BERTopic's own "embeddings" outlier /
scaling strategy, so every utterance gets a real topic (no 41% outlier dump).

    python -m analysis.topic_clusters --k 40 --sample 150000

Outputs:
    out/topic_clusters_k{k}.csv               id, corpus_count, label, top words
    out/cache/topic_assignment_k{k}.npy       int16 topic id per corpus row (aligned)

Downstream: analysis.endogenous_hawkes loads these via load_topics() and fits one
visibility-marked Hawkes per topic, exactly like the LLM-coded behaviours.
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np
import pandas as pd

from analysis import load
from analysis.meme_mining.llm_coded import behaviour_propagate as BP

OUT, CACHE = "out", "out/cache"
EMB_MODEL = "all-MiniLM-L6-v2"

# Ubiquitous platform vocabulary that would otherwise dominate every label.
EXTRA_STOP = ["molt", "molty", "moltys", "moltbook", "claw", "claws", "clawd",
              "agent", "agents", "ai", "just", "like", "https", "http", "com",
              "www", "im", "youre", "dont", "thats", "ive"]

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_CODEBLOCK = re.compile(r"`{1,3}[^`]*`{1,3}")


def _clean(text: str) -> str:
    text = _URL_RE.sub(" ", text)
    text = _CODEBLOCK.sub(" ", text)
    return " ".join(text.split())


def _assign_full(emb_path: str, centroids: np.ndarray, post_mask: np.ndarray,
                 chunk: int = 200_000) -> np.ndarray:
    """Nearest-centroid topic COLUMN for every POST row; -1 for comments.
    Topics are a post-level (feed) phenomenon, so comments are left unassigned."""
    emb = np.load(emb_path, mmap_mode="r")
    n = emb.shape[0]
    out = np.full(n, -1, dtype=np.int16)
    cT = centroids.T.astype(np.float32)          # (384, K)
    for i in range(0, n, chunk):
        pm = post_mask[i:i + chunk]
        if not pm.any():
            continue
        block = np.asarray(emb[i:i + chunk], dtype=np.float32)[pm]
        out[i:i + chunk][pm] = (block @ cT).argmax(axis=1).astype(np.int16)
    return out


def _label(words: list[str]) -> str:
    toks = [w for w in words if w.lower() not in EXTRA_STOP and len(w) >= 3][:4]
    return " / ".join(toks) if toks else "(unlabelled)"


def fit(k: int = 40, sample: int = 150_000, seed: int = 0, min_len: int = 40):
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
    from umap import UMAP

    corp = load.corpus()
    emb_path = os.path.join(CACHE, f"corpus_emb_{EMB_MODEL}.npy")
    emb = BP.embed_corpus(corp)                    # (2.1M, 384) normalized, cached

    # --- POSTS ONLY: topics are a feed (post-level) phenomenon. Comments (86% of
    # the corpus) are reply chatter and must not define or receive topics.
    post_mask = (corp["source"] == "post").to_numpy()
    print(f"      posts={int(post_mask.sum()):,} of {len(corp):,} rows "
          f"(comments excluded from fit AND assignment)")

    # --- fit sample: posts only, length filter, clean, EXACT dedup
    text = corp["text"].fillna("")
    keep = (text.str.len() >= min_len).to_numpy() & post_mask
    idx_all = np.where(keep)[0]
    cleaned = text.iloc[idx_all].map(_clean)
    ok = cleaned.str.len() >= 20
    idx_all, cleaned = idx_all[ok.to_numpy()], cleaned[ok.to_numpy()]
    # first occurrence of each exact cleaned text
    seen_first = ~cleaned.duplicated()
    idx_uniq = idx_all[seen_first.to_numpy()]
    docs_uniq = cleaned[seen_first.to_numpy()].tolist()
    print(f"[1/5] fit pool: {len(idx_all):,} len-ok -> {len(idx_uniq):,} exact-unique")
    rng = np.random.default_rng(seed)
    take = min(sample, len(idx_uniq))
    sel = rng.choice(len(idx_uniq), size=take, replace=False)
    fit_idx = idx_uniq[sel]
    fit_docs = [docs_uniq[i] for i in sel]
    fit_emb = np.asarray(emb[fit_idx], dtype=np.float32)
    print(f"      fitting on {take:,} exact-unique utterances")

    stops = list(ENGLISH_STOP_WORDS | set(EXTRA_STOP))
    vectorizer = CountVectorizer(stop_words=stops, min_df=3, ngram_range=(1, 2),
                                 token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]{2,}\b")
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                      metric="cosine", random_state=seed)
    print(f"[2/5] BERTopic fit (reduce to ~{k} topics) ...")
    tm = BERTopic(vectorizer_model=vectorizer, umap_model=umap_model,
                  min_topic_size=max(50, take // 1500), nr_topics=k + 1,
                  calculate_probabilities=False, verbose=True)
    topics, _ = tm.fit_transform(fit_docs, fit_emb)
    topics = np.asarray(topics)
    info = tm.get_topic_info()
    real_ids = [t for t in info["Topic"].tolist() if t >= 0]
    print(f"      {len(real_ids)} topics ({int((topics == -1).sum()):,} sample outliers)")

    # --- topic centroids in ORIGINAL 384-d space (mean of confident members)
    cents, ids, words_by = [], [], {}
    for t in real_ids:
        members = fit_emb[topics == t]
        if len(members) == 0:
            continue
        c = members.mean(axis=0)
        c /= (np.linalg.norm(c) + 1e-9)
        cents.append(c); ids.append(t)
        words_by[t] = [w for w, _ in tm.get_topic(t)]
    centroids = np.vstack(cents).astype(np.float32)
    print(f"[3/5] assigning {int(post_mask.sum()):,} posts by nearest centroid "
          f"(comments -> -1) ...")
    col = _assign_full(emb_path, centroids, post_mask)   # col index, or -1 for comments
    ids_arr = np.array(ids, dtype=np.int16)
    assign = np.where(col >= 0, ids_arr[np.clip(col, 0, None)], -1).astype(np.int16)

    # --- persist assignment (aligned to corpus row order)
    os.makedirs(CACHE, exist_ok=True)
    apath = os.path.join(CACHE, f"topic_assignment_k{k}.npy")
    np.save(apath, assign)

    # --- topic table with FULL-corpus counts + labels
    counts = pd.Series(assign).value_counts()
    rows = []
    for t in ids:
        rows.append(dict(topic=int(t), corpus_count=int(counts.get(t, 0)),
                         label=_label(words_by[t]),
                         top_words=", ".join(words_by[t][:8])))
    tbl = pd.DataFrame(rows).sort_values("corpus_count", ascending=False).reset_index(drop=True)
    tpath = os.path.join(OUT, f"topic_clusters_k{k}.csv")
    tbl.to_csv(tpath, index=False)
    print(f"[4/5] wrote {tpath} and {apath}")
    print(f"[5/5] top topics by corpus share:")
    print(tbl.head(20).to_string(index=False))
    return tbl, assign


def load_topics(corp: pd.DataFrame, k: int = 40):
    """(labels, assignment) for the Hawkes. labels: list of (topic_id, label);
    assignment: int16 array (topic id per corp row). Refits if cache is missing."""
    apath = os.path.join(CACHE, f"topic_assignment_k{k}.npy")
    tpath = os.path.join(OUT, f"topic_clusters_k{k}.csv")
    if not (os.path.exists(apath) and os.path.exists(tpath)):
        fit(k=k)
    assign = np.load(apath)
    if len(assign) != len(corp):
        raise ValueError(f"topic assignment ({len(assign)}) != corpus ({len(corp)}); refit")
    tbl = pd.read_csv(tpath)
    labels = list(zip(tbl["topic"].astype(int), tbl["label"]))
    return labels, assign


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--sample", type=int, default=150_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-len", type=int, default=40)
    args = ap.parse_args()
    fit(k=args.k, sample=args.sample, seed=args.seed, min_len=args.min_len)
