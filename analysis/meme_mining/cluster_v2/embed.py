"""Step-3 meme mining, stage 1: strong POST embeddings.

Replaces the MiniLM-384 embeddings used by the old BERTopic pipeline. We embed
POSTS ONLY (the feed ranks posts; state-mediated contagion is a post-level
phenomenon) with a strong instruction-tuned sentence encoder (default
BAAI/bge-large-en-v1.5, 1024-d), on MPS. Embeddings are L2-normalised so cosine
== dot, cached aligned to load.load_posts() row order.

    python -m analysis.meme_mining.cluster_v2.embed                 # bge-large
    python -m analysis.meme_mining.cluster_v2.embed --model sentence-transformers/all-mpnet-base-v2

Output: out/cache/postemb_<model-tag>.npy  (float32, [n_posts, d], unit-norm)
        out/cache/postemb_<model-tag>.meta.json
"""
from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np

from analysis import load

CACHE = "out/cache"
_URL = re.compile(r"https?://\S+|www\.\S+")
_CODE = re.compile(r"`{1,3}[^`]*`{1,3}")

# bge/e5 want a query/passage instruction; for symmetric clustering bge uses a bare
# "represent this sentence for retrieval" style prefix. Empty for mpnet.
INSTRUCTION = {
    "BAAI/bge-large-en-v1.5": "Represent this social media post for clustering: ",
    "BAAI/bge-base-en-v1.5": "Represent this social media post for clustering: ",
    "intfloat/e5-large-v2": "query: ",
}


def _clean(t: str) -> str:
    t = _URL.sub(" ", str(t))
    t = _CODE.sub(" ", t)
    return " ".join(t.split())


def tag_of(model: str) -> str:
    return model.split("/")[-1]


def embed_posts(model="BAAI/bge-large-en-v1.5", batch=256, max_chars=400, limit=None,
                max_seq_len=192, chunk=10000):
    """RESUMABLE chunked embedding. Each chunk of `chunk` posts is embedded and saved
    to postemb_<tag>_chunks/chunk_NNNNN.npy; a re-run skips chunks already on disk, so
    an interrupted long job (bge-large ~ hours) continues where it stopped. When all
    chunks exist they are concatenated into the final postemb_<tag>.npy."""
    os.makedirs(CACHE, exist_ok=True)
    tag = tag_of(model)
    out_path = os.path.join(CACHE, f"postemb_{tag}.npy")
    meta_path = os.path.join(CACHE, f"postemb_{tag}.meta.json")
    cdir = os.path.join(CACHE, f"postemb_{tag}_chunks")
    os.makedirs(cdir, exist_ok=True)
    posts = load.load_posts()
    texts = posts["text"].fillna("").map(_clean).str.slice(0, max_chars).tolist()
    if limit:
        texts = texts[:limit]
    n = len(texts)
    if os.path.exists(out_path):
        emb = np.load(out_path, mmap_mode="r")
        if emb.shape[0] == n:
            print(f"cached: {out_path} {emb.shape}", flush=True)
            return out_path

    starts = list(range(0, n, chunk))
    prefix = INSTRUCTION.get(model, "")

    def cpath(ci):
        return os.path.join(cdir, f"chunk_{ci:05d}.npy")

    def expected_rows(ci):
        return min(chunk, n - starts[ci])

    todo = [ci for ci in range(len(starts))
            if not (os.path.exists(cpath(ci))
                    and np.load(cpath(ci), mmap_mode="r").shape[0] == expected_rows(ci))]
    print(f"{n} posts, {len(starts)} chunks; {len(todo)} remaining (resuming)", flush=True)

    if todo:
        from sentence_transformers import SentenceTransformer
        import torch
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"loading {model} on {dev} (max_seq_len={max_seq_len}) ...", flush=True)
        st = SentenceTransformer(model, device=dev)
        st.max_seq_length = max_seq_len
        for k, ci in enumerate(todo):
            s = starts[ci]
            block = texts[s:s + chunk]
            if prefix:
                block = [prefix + t for t in block]
            e = st.encode(block, batch_size=batch, normalize_embeddings=True,
                          show_progress_bar=False, convert_to_numpy=True).astype("float32")
            np.save(cpath(ci), e)
            print(f"  chunk {ci+1}/{len(starts)} done ({k+1}/{len(todo)} this run)", flush=True)

    emb = np.concatenate([np.load(cpath(ci)) for ci in range(len(starts))], axis=0)
    assert emb.shape[0] == n, f"assembled {emb.shape[0]} != {n}"
    np.save(out_path, emb)
    json.dump({"model": model, "n": int(emb.shape[0]), "dim": int(emb.shape[1]),
               "normalized": True, "max_chars": max_chars, "max_seq_len": max_seq_len,
               "instruction": prefix}, open(meta_path, "w"), indent=2)
    print(f"wrote {out_path} {emb.shape}", flush=True)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="BAAI/bge-large-en-v1.5")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-seq-len", type=int, default=192)
    ap.add_argument("--chunk", type=int, default=10000)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    embed_posts(model=args.model, batch=args.batch, limit=args.limit,
                max_seq_len=args.max_seq_len, chunk=args.chunk)
