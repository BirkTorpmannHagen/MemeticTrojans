"""Load and normalize the Moltbook parquet files.

Provides the post / comment tables and a unified per-utterance corpus for meme
scanning and the state-mediated (feed-visibility) analyses. Moltbook is modelled
state-mediated (feed rank), never as a comment/reply interaction graph; the only
edge substrate in the project is the SNAP ego-Twitter follower graph
(sandbox/reach_edge.py). Intermediate artifacts are cached so reruns are fast.
"""

from __future__ import annotations

import os
from functools import lru_cache

import pandas as pd

RAW_DIR = "data_raw"
CACHE_DIR = "out/cache"

POSTS_PARQUET = os.path.join(RAW_DIR, "posts.parquet")
COMMENTS_PARQUET = os.path.join(RAW_DIR, "comments.parquet")


def _ensure_cache() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)


@lru_cache(maxsize=1)
def load_posts() -> pd.DataFrame:
    df = pd.read_parquet(POSTS_PARQUET)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df["date"] = df["created_at"].dt.date
    # Combined text field for meme scanning (title carries a lot of the signal).
    df["text"] = (df["title"].fillna("") + "\n" + df["content"].fillna("")).str.strip()
    return df


@lru_cache(maxsize=1)
def load_comments() -> pd.DataFrame:
    df = pd.read_parquet(COMMENTS_PARQUET)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df["date"] = df["created_at"].dt.date
    df["text"] = df["content"].fillna("").str.strip()
    return df


def corpus() -> pd.DataFrame:
    """Unified author/text/time/date frame over posts + top-level content.

    One row per authored utterance, tagged by source. Used for meme scanning so
    that 'author spread' counts a meme once per author-utterance across the
    whole platform.
    """
    posts = load_posts()
    comments = load_comments()
    p = posts[["author_name", "text", "created_at", "date", "submolt_name"]].copy()
    p["source"] = "post"
    cm = comments[["author_name", "text", "created_at", "date"]].copy()
    cm["submolt_name"] = pd.NA
    cm["source"] = "comment"
    out = pd.concat([p, cm], ignore_index=True)
    out = out.dropna(subset=["author_name", "created_at"])  # NaT breaks time ops
    out = out[out["text"].str.len() > 0].reset_index(drop=True)
    return out
