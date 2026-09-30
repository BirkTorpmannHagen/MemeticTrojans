"""Construct realistic Moltbook heartbeat feeds from the real post corpus.

Replaces the synthetic distractor pool in run_rank_exposure with actual Moltbook
posts (real titles/bodies/upvotes/timestamps), ranked exactly as the platform ranks:
karma + recency (Forum._score = recency + 0.2*upvotes). A "heartbeat feed" is the
top-K scored posts in a recency window ending at a randomly sampled cycle time T.

This makes the rank-exposure assay faithful: the distractors the payload competes
against are the real feed an agent would actually be served, with real social-proof
numbers and real content, not a hand-written pool.

SAFETY: read-only. Posts come from the local cached corpus; nothing is fetched or
posted. The payload we splice in is still an inert placeholder (see run_rank_exposure).
"""
from __future__ import annotations

import functools
import numpy as np
import pandas as pd

CACHE = "out/exposure/cache/posts_trimmed.parquet"


@functools.lru_cache(maxsize=1)
def load_posts(path: str = CACHE) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if not pd.api.types.is_datetime64_any_dtype(df.created_at):
        df["created_at"] = pd.to_datetime(df.created_at, utc=True)
    return df.sort_values("created_at").reset_index(drop=True)


def _score(upvotes, hours_ago, half_life_h: float = 3.0):
    """Platform ranking: recency + 0.2*upvotes, matching Forum._score.
    recency = 1/(1+Δ) with Δ in half-life units so a ~half-life-old post decays like
    one forum round. Upvotes then dominate once a post is a few units old (karma loop)."""
    recency = 1.0 / (1.0 + hours_ago / half_life_h)
    return recency + 0.2 * upvotes


def sample_feed(k: int, seed: int, window_hours: float = 6.0,
                half_life_h: float = 3.0, path: str = CACHE, sort: str = "hot") -> pd.DataFrame:
    """Return K real posts as one heartbeat feed: pick a random cycle time T, take posts
    created in (T-window, T], rank, return the top-K (rank order).

    sort="hot": rank by recency+0.2*upvotes (platform hot, default).
    sort="new": rank by recency only (newest-first) — the chronological `new` surface, no karma.
    Deterministic in `seed`. The returned frame carries a `feed_upvotes` column (the
    upvotes shown in-feed) and is ordered rank 0..K-1 (best first)."""
    df = load_posts(path)
    rng = np.random.default_rng(seed)
    lo = df.created_at.quantile(0.05)   # avoid the sparse cold-start / tail edges
    hi = df.created_at.quantile(0.95)
    span = (hi - lo).total_seconds()
    for _ in range(200):                # rejection-sample a T with a full window of posts
        t = lo + pd.Timedelta(seconds=rng.uniform(0, span))
        w = df[(df.created_at > t - pd.Timedelta(hours=window_hours)) & (df.created_at <= t)]
        if len(w) >= k:
            break
    if sort == "new":                   # chronological: newest posts first, karma ignored
        top = w.sort_values("created_at", ascending=False).head(k).reset_index(drop=True)
    else:                               # hot: recency + 0.2*upvotes
        hours_ago = (t - w.created_at).dt.total_seconds() / 3600.0
        scored = w.assign(_s=_score(w.upvotes.values, hours_ago.values, half_life_h))
        top = scored.sort_values("_s", ascending=False).head(k).reset_index(drop=True)
    top = top.rename(columns={"upvotes": "feed_upvotes"})
    return top


def render_slot(disp_id: int, submolt: str, author: str, title: str, body: str,
                upv: int, show_upvotes: bool, body_chars: int = 220) -> str:
    head = f"[post_id={disp_id} | m/{submolt} | by @{author}"
    head += f" | {upv} upvotes]" if show_upvotes else "]"
    body = (body or "").strip().replace("\n", " ")
    if len(body) > body_chars:
        body = body[:body_chars].rstrip() + "…"
    t = (title or "").strip()
    return f"{head}\n{t}\n{body}" if t else f"{head}\n{body}"
