"""Distill LLM behaviour labels to the full corpus (weak-supervision classifier).

The LLM labeled ~30k utterances (analysis/meme_mining/llm_coded/llm_labeling.py). Here we:

  1. embed the FULL corpus locally (sentence-transformers, cached);
  2. train a per-category classifier on the LLM-labeled sample (embeddings ->
     multi-label behaviour), and report held-out precision/recall/F1 so we know
     how trustworthy the propagated labels are;
  3. predict every utterance's behaviours over all 2.1M.

`embed_corpus` / `train_predict` / `CATS` are consumed by the state-mediated Hawkes
behaviour mode (analysis/estimate_tau.py, analysis/endogenous_hawkes.py, run with
`--behaviours`): each predicted behaviour mask is fed into the feed-visibility Hawkes
engine (R_endo / phi_endo), the same estimator used for lexical memes. There is no
comment/reply interaction graph — Moltbook contagion is measured state-mediated.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from analysis.meme_mining.llm_coded.llm_labeling import CATEGORIES, LABELS_PATH

CACHE = "out/cache"
CATS = [c for c, _ in CATEGORIES if c != "none"]
EMB_MODEL = "all-MiniLM-L6-v2"


def embed_corpus(corp: pd.DataFrame, model_name: str = EMB_MODEL) -> np.ndarray:
    """Embed every utterance (cached, aligned to corp row order)."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"corpus_emb_{model_name.replace('/', '_')}.npy")
    if os.path.exists(path):
        emb = np.load(path)
        if emb.shape[0] == len(corp):
            print(f"      loaded cached embeddings {emb.shape}")
            return emb
        print("      cache size mismatch; re-embedding")
    from sentence_transformers import SentenceTransformer
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"      embedding {len(corp):,} utterances with {model_name} on {device} ...")
    m = SentenceTransformer(model_name, device=device)
    # Behaviour/intent signal sits early in the text; cap sequence length so short
    # utterances (the vast majority) don't pay for the 256-token default. ~2x faster.
    m.max_seq_length = 128
    emb = m.encode(corp["text"].tolist(), batch_size=512, show_progress_bar=True,
                   convert_to_numpy=True, normalize_embeddings=True)
    np.save(path, emb)
    return emb


def _multihot(labels_col: pd.Series) -> np.ndarray:
    idx = {c: i for i, c in enumerate(CATS)}
    Y = np.zeros((len(labels_col), len(CATS)), dtype=np.int8)
    for i, labs in enumerate(labels_col.values):
        if labs is None:
            continue
        for l in labs:
            j = idx.get(l)
            if j is not None:
                Y[i, j] = 1
    return Y


def train_predict(emb_full, corp, seed=0, threshold=0.5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_fscore_support

    lab = pd.read_parquet(LABELS_PATH)
    lab = lab[lab["doc_id"] < len(corp)]
    Xtr = emb_full[lab["doc_id"].to_numpy()]
    Y = _multihot(lab["labels"])
    print(f"      training on {len(lab):,} labeled docs; positives/category:")
    for c, n in zip(CATS, Y.sum(0)):
        print(f"        {c:26s} {int(n):5d}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(lab))
    cut = int(0.8 * len(lab))
    tr, te = perm[:cut], perm[cut:]

    pred_full = np.zeros((emb_full.shape[0], len(CATS)), dtype=bool)
    evals = []
    for j, c in enumerate(CATS):
        y = Y[:, j]
        if y[tr].sum() < 5:
            evals.append({"category": c, "positives": int(y.sum()),
                          "threshold": np.nan, "precision": np.nan,
                          "recall": np.nan, "f1": np.nan, "note": "too few positives"})
            continue
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        clf.fit(Xtr[tr], y[tr])
        proba = clf.predict_proba(Xtr[te])[:, 1]
        # Pick the decision threshold that maximizes held-out F1. Balanced weights
        # push recall up at 0.5; tuning restores precision so a rare category
        # (money_making) stays sharp instead of re-blurring to keyword-level.
        best_t, best_f1, best_pr, best_rc = 0.5, -1.0, 0.0, 0.0
        for t in np.linspace(0.2, 0.9, 29):
            p = proba >= t
            pr, rc, f1, _ = precision_recall_fscore_support(
                y[te], p, average="binary", zero_division=0)
            if f1 > best_f1:
                best_t, best_f1, best_pr, best_rc = t, f1, pr, rc
        evals.append({"category": c, "positives": int(y.sum()),
                      "threshold": round(best_t, 3), "precision": round(best_pr, 3),
                      "recall": round(best_rc, 3), "f1": round(best_f1, 3), "note": ""})
        # refit on all labeled data, predict full corpus at the tuned threshold
        clf.fit(Xtr, y)
        pred_full[:, j] = clf.predict_proba(emb_full)[:, 1] >= best_t
    return pred_full, pd.DataFrame(evals)
