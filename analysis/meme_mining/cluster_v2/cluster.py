"""Step-3 meme mining, stage 2: self-determining clustering + principled assessment.

Data-driven replacement for the hand-forced k=40 BERTopic reduction. Pipeline:
  embeddings (bge-large, stage 1) -> UMAP (cosine, low-dim) -> HDBSCAN (density,
  finds the cluster COUNT itself via min_cluster_size; has an explicit noise class).

We do NOT fix the number of clusters. Instead we SWEEP min_cluster_size and pick
the configuration by an ensemble of standard cluster-quality criteria, none of which
requires a target k:
  - coverage         : 1 - noise_fraction (HDBSCAN leaves low-density points as -1)
  - n_clusters       : how many topics emerge
  - DBCV             : density-based cluster validity (HDBSCAN relative_validity_),
                       the validity index designed for density clusters (Moulavi 2014)
  - silhouette       : mean silhouette on non-noise points (embedding space)
  - NPMI coherence   : mean pairwise NPMI of each cluster's top c-TF-IDF words over
                       the post corpus (Bouma 2009) -- the standard topic-coherence
                       metric (we implement it directly; gensim is absent)
  - stability (ARI)  : agreement of assignments across UMAP seeds (only for the pick)

    python -m analysis.meme_mining.cluster_v2.cluster --emb out/cache/postemb_bge-large-en-v1.5.npy

Outputs (out/cluster_v2/):
  sweep_metrics.csv         one row per min_cluster_size with all criteria
  assignment_mcs<N>.npy     int32 cluster id per POST row (-1 = noise), for the pick
  labels_mcs<N>.csv         cluster id, size, top c-TF-IDF words (label)
"""
from __future__ import annotations

import argparse
import os
import re

import numpy as np
import pandas as pd

from analysis import load

OUT = "out/cluster_v2"
_URL = re.compile(r"https?://\S+|www\.\S+")
_CODE = re.compile(r"`{1,3}[^`]*`{1,3}")
EXTRA_STOP = {"molt", "molty", "moltys", "moltbook", "claw", "claws", "clawd", "clawde",
              "agent", "agents", "ai", "just", "like", "https", "http", "com", "www",
              "im", "youre", "dont", "thats", "ive", "th", "de", "la"}


def _clean(t):
    t = _URL.sub(" ", str(t)); t = _CODE.sub(" ", t)
    return " ".join(t.split())


# --------------------------------------------------------------------------- #
# UMAP + HDBSCAN                                                               #
# --------------------------------------------------------------------------- #
def umap_reduce(emb, n_components=5, n_neighbors=15, seed=42):
    import umap
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                        min_dist=0.0, metric="cosine", random_state=seed, verbose=True)
    return reducer.fit_transform(emb)


def hdbscan_cluster(X, min_cluster_size, min_samples=10):
    import hdbscan
    cl = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                         metric="euclidean", cluster_selection_method="eom",
                         gen_min_span_tree=True, core_dist_n_jobs=-1)
    labels = cl.fit_predict(X)
    dbcv = float(getattr(cl, "relative_validity_", np.nan))
    return labels, dbcv


# --------------------------------------------------------------------------- #
# c-TF-IDF cluster labels                                                      #
# --------------------------------------------------------------------------- #
def ctfidf_labels(texts, labels, topn=8):
    from sklearn.feature_extraction.text import CountVectorizer
    ids = sorted(i for i in set(labels) if i >= 0)
    docs = {i: [] for i in ids}
    for t, l in zip(texts, labels):
        if l >= 0:
            docs[l].append(t)
    joined = [" ".join(docs[i]) for i in ids]
    # each cluster is ONE document here, so min_df must be 1 (a term distinctive to a
    # single cluster is exactly what c-TF-IDF should surface); idf down-weights the rest
    cv = CountVectorizer(stop_words="english", min_df=1, ngram_range=(1, 2),
                         token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    tf = cv.fit_transform(joined).toarray().astype("float64")
    vocab = np.array(cv.get_feature_names_out())
    # c-TF-IDF: tf normalised per class * log(1 + N_avg / df_class)
    tf_norm = tf / np.maximum(tf.sum(axis=1, keepdims=True), 1)
    df = (tf > 0).sum(axis=0)
    idf = np.log(1.0 + tf.shape[0] / np.maximum(df, 1))
    ctfidf = tf_norm * idf
    labels_out = {}
    for r, i in enumerate(ids):
        order = np.argsort(ctfidf[r])[::-1]
        words = [vocab[j] for j in order if vocab[j] not in EXTRA_STOP][:topn]
        labels_out[i] = words
    return labels_out


# --------------------------------------------------------------------------- #
# NPMI coherence (Bouma 2009) over the post corpus                             #
# --------------------------------------------------------------------------- #
def npmi_coherence(cluster_words, texts, topk=8, sample=40000, seed=0):
    """Mean pairwise NPMI of each cluster's top words, averaged over clusters.
    Co-occurrence is at the post level over a sampled subset for speed."""
    from sklearn.feature_extraction.text import CountVectorizer
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(texts), min(sample, len(texts)), replace=False)
    sub = [texts[i] for i in idx]
    vocab = sorted({w for ws in cluster_words.values() for w in ws[:topk]})
    if len(vocab) < 2:
        return float("nan")
    cv = CountVectorizer(vocabulary=vocab, binary=True,
                         token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b", ngram_range=(1, 2))
    M = cv.fit_transform(sub)                      # [n_docs, V] binary
    N = M.shape[0]
    Pw = np.asarray(M.sum(axis=0)).ravel() / N     # P(w)
    Co = (M.T @ M).toarray() / N                   # P(w_i, w_j)
    vi = {w: k for k, w in enumerate(vocab)}
    per = []
    for ws in cluster_words.values():
        top = [w for w in ws[:topk] if w in vi]
        pn = []
        for a in range(len(top)):
            for b in range(a + 1, len(top)):
                i, j = vi[top[a]], vi[top[b]]
                pij = Co[i, j]
                if pij <= 0:
                    pn.append(-1.0); continue
                npmi = np.log(pij / (Pw[i] * Pw[j])) / (-np.log(pij))
                pn.append(float(npmi))
        if pn:
            per.append(np.mean(pn))
    return float(np.mean(per)) if per else float("nan")


def silhouette_nonnoise(X, labels, sample=30000, seed=0):
    from sklearn.metrics import silhouette_score
    m = labels >= 0
    if m.sum() < 3 or len(set(labels[m])) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    idx = np.where(m)[0]
    if len(idx) > sample:
        idx = rng.choice(idx, sample, replace=False)
    return float(silhouette_score(X[idx], labels[idx]))


def run(emb_path, min_cluster_sizes=(50, 100, 200, 400, 800), min_samples=10,
        n_components=5, seed=42, stability_seeds=(1, 2, 3)):
    os.makedirs(OUT, exist_ok=True)
    posts = load.load_posts()
    texts = posts["text"].fillna("").map(_clean).tolist()
    emb = np.load(emb_path, mmap_mode="r")
    assert emb.shape[0] == len(texts), f"emb {emb.shape[0]} != posts {len(texts)}"
    emb = np.asarray(emb, dtype="float32")

    print(f"UMAP {emb.shape} -> {n_components}d ...", flush=True)
    X = umap_reduce(emb, n_components=n_components, seed=seed)

    rows, assignments, labelmaps = [], {}, {}
    for mcs in min_cluster_sizes:
        labels, dbcv = hdbscan_cluster(X, mcs, min_samples)
        ids = [i for i in set(labels) if i >= 0]
        noise = float(np.mean(labels < 0))
        lab = ctfidf_labels(texts, labels)
        npmi = npmi_coherence(lab, texts)
        sil = silhouette_nonnoise(X, labels)
        rows.append({"min_cluster_size": mcs, "n_clusters": len(ids),
                     "coverage": round(1 - noise, 3), "dbcv": round(dbcv, 3),
                     "silhouette": round(sil, 3), "npmi": round(npmi, 4)})
        assignments[mcs] = labels
        labelmaps[mcs] = lab
        print(f"  mcs={mcs:4d}  k={len(ids):3d}  cov={1-noise:.2f}  "
              f"dbcv={dbcv:.3f}  sil={sil:.3f}  npmi={npmi:.4f}", flush=True)

    met = pd.DataFrame(rows)
    met.to_csv(os.path.join(OUT, "sweep_metrics.csv"), index=False)

    # pick: rank-normalise dbcv, silhouette, npmi, coverage; prefer higher; require k>=8
    cand = met[met["n_clusters"] >= 8].copy()
    if cand.empty:
        cand = met.copy()
    for c in ["dbcv", "silhouette", "npmi", "coverage"]:
        cand[c + "_r"] = cand[c].rank()
    cand["score"] = cand[[c + "_r" for c in ["dbcv", "silhouette", "npmi", "coverage"]]].mean(axis=1)
    best = int(cand.sort_values("score").iloc[-1]["min_cluster_size"])
    print(f"\nPICK: min_cluster_size={best}", flush=True)

    # stability: ARI of the pick across UMAP seeds
    from sklearn.metrics import adjusted_rand_score
    base = assignments[best]
    aris = []
    for s in stability_seeds:
        Xs = umap_reduce(emb, n_components=n_components, seed=s)
        ls, _ = hdbscan_cluster(Xs, best, min_samples)
        aris.append(adjusted_rand_score(base, ls))
    print(f"stability (mean ARI over UMAP seeds {stability_seeds}): {np.mean(aris):.3f}", flush=True)

    np.save(os.path.join(OUT, f"assignment_mcs{best}.npy"), base.astype("int32"))
    lab = labelmaps[best]
    lr = [{"cluster": i, "size": int(np.sum(base == i)),
           "label": " / ".join(lab[i][:4]), "top_words": " ".join(lab[i])}
          for i in sorted(lab)]
    pd.DataFrame(lr).sort_values("size", ascending=False).to_csv(
        os.path.join(OUT, f"labels_mcs{best}.csv"), index=False)
    met.to_csv(os.path.join(OUT, "sweep_metrics.csv"), index=False)
    print(f"wrote assignment_mcs{best}.npy + labels_mcs{best}.csv ({len(lr)} clusters, "
          f"mean ARI {np.mean(aris):.2f})", flush=True)
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="out/cache/postemb_bge-large-en-v1.5.npy")
    ap.add_argument("--mcs", type=int, nargs="+", default=[50, 100, 200, 400, 800])
    ap.add_argument("--min-samples", type=int, default=10)
    ap.add_argument("--components", type=int, default=5)
    args = ap.parse_args()
    run(args.emb, min_cluster_sizes=tuple(args.mcs), min_samples=args.min_samples,
        n_components=args.components)
