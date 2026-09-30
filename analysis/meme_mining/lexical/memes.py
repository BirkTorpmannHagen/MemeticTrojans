"""Meme identification by cross-author lexical reuse.

A "meme" here is a transmissible textual unit that many *distinct* agents
reproduce. We mine four candidate classes and score each by how widely it
spread across the author population (not just how often it occurred — that
would reward one chatty bot or a platform template):

  * hashtags        (#molt, #agentinternet, ...)
  * emoji           (🦞, 🜁, ...)
  * link domains    (agentmarket.cloud, github.com/..., ...)
  * phrase n-grams  (4- and 5-word slogans / catchphrases)

For every candidate we record: distinct-author spread, total occurrences,
first-seen timestamp, active-day span, and the single-author concentration
(used to filter boilerplate that one account repeats).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

import pandas as pd

# Token classes -------------------------------------------------------------
_WORD = re.compile(r"[a-z0-9']+")
_HASHTAG = re.compile(r"#\w{2,40}")
_URL = re.compile(r"https?://([^\s/]+)")
# Emoji / pictographic ranges (covers the lobster, alchemical, symbol blocks).
_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F004\U0001F0CF"
    "\U00002190-\U000021FF\U00002B00-\U00002BFF\U0001F1E6-\U0001F1FF\U0001F900-\U0001F9FF]"
)


def _explode_tokens(df: pd.DataFrame, pattern: re.Pattern, lower: bool) -> pd.DataFrame:
    """Return long-form (token, author, date) rows for a regex token class,
    counting each token at most once per utterance (so author-spread is by
    author-utterance, not by repetition within a single post)."""
    rows_tok, rows_au, rows_dt = [], [], []
    for txt, au, dt in zip(df["text"].values, df["author_name"].values, df["date"].values):
        found = pattern.findall(txt.lower() if lower else txt)
        if not found:
            continue
        for tok in set(found):
            rows_tok.append(tok)
            rows_au.append(au)
            rows_dt.append(str(dt))  # ISO string sorts correctly and avoids
            #                          object-dtype groupby.min() errors
    return pd.DataFrame({"token": rows_tok, "author": rows_au, "date": rows_dt})


def _aggregate(long: pd.DataFrame, kind: str, min_authors: int) -> pd.DataFrame:
    if long.empty:
        return pd.DataFrame()
    g = long.groupby("token")
    stats = pd.DataFrame(
        {
            "kind": kind,
            "occurrences": g.size(),
            "distinct_authors": g["author"].nunique(),
            "first_seen": g["date"].min(),
            "last_seen": g["date"].max(),
        }
    )
    # Single-author concentration: occurrences by the most prolific user / total.
    top_share = g["author"].agg(lambda s: s.value_counts().iloc[0] / len(s))
    stats["top_author_share"] = top_share
    stats["active_days"] = (
        pd.to_datetime(stats["last_seen"]) - pd.to_datetime(stats["first_seen"])
    ).dt.days + 1
    stats = stats[stats["distinct_authors"] >= min_authors]
    return stats.reset_index().rename(columns={"token": "meme"})


def scan_token_memes(corpus: pd.DataFrame, min_authors: int = 25) -> pd.DataFrame:
    """Hashtags, emoji and link-domains across the full corpus."""
    parts = []
    parts.append(_aggregate(_explode_tokens(corpus, _HASHTAG, lower=True),
                            "hashtag", min_authors))
    parts.append(_aggregate(_explode_tokens(corpus, _EMOJI, lower=False),
                            "emoji", min_authors))
    parts.append(_aggregate(_explode_tokens(corpus, _URL, lower=True),
                            "domain", min_authors))
    out = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    return out


# Phrase n-gram mining ------------------------------------------------------

def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def mine_phrase_candidates(
    texts, ngram_sizes=(4, 5), min_freq: int = 20, prune_every: int = 300_000
) -> set[tuple]:
    """Pass 1: frequent n-grams. Memory is bounded by periodically dropping
    singletons — true memes recur, so this only loses genuine one-offs."""
    cnt: Counter = Counter()
    for i, txt in enumerate(texts):
        toks = _tokens(txt)
        for n in ngram_sizes:
            for j in range(len(toks) - n + 1):
                cnt[tuple(toks[j : j + n])] += 1
        if (i + 1) % prune_every == 0:
            cnt = Counter({k: v for k, v in cnt.items() if v > 1})
    return {k for k, v in cnt.items() if v >= min_freq}


def phrase_stats(texts, authors, dates, candidates: set[tuple]) -> pd.DataFrame:
    """Pass 2: distinct-author spread, occurrences, and time span for each
    candidate phrase."""
    authors_by: dict[tuple, set] = defaultdict(set)
    occ: Counter = Counter()
    first: dict[tuple, object] = {}
    last: dict[tuple, object] = {}
    by_top_author: dict[tuple, Counter] = defaultdict(Counter)
    sizes = sorted({len(c) for c in candidates}) or [4, 5]

    for txt, au, dt in zip(texts, authors, dates):
        toks = _tokens(txt)
        hits = set()
        for n in sizes:
            for j in range(len(toks) - n + 1):
                g = tuple(toks[j : j + n])
                if g in candidates:
                    hits.add(g)
        for g in hits:
            authors_by[g].add(au)
            occ[g] += 1
            by_top_author[g][au] += 1
            if g not in first or dt < first[g]:
                first[g] = dt
            if g not in last or dt > last[g]:
                last[g] = dt

    rows = []
    for g in candidates:
        if g not in occ:
            continue
        ta = by_top_author[g].most_common(1)[0][1]
        rows.append(
            {
                "kind": "phrase",
                "meme": " ".join(g),
                "_tuple": g,
                "occurrences": occ[g],
                "distinct_authors": len(authors_by[g]),
                "first_seen": first[g],
                "last_seen": last[g],
                "top_author_share": ta / occ[g],
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["active_days"] = (
        pd.to_datetime(df["last_seen"]) - pd.to_datetime(df["first_seen"])
    ).dt.days + 1
    return df


def collapse_subphrases(phrases: pd.DataFrame, cap: int = 1500) -> pd.DataFrame:
    """Drop a shorter phrase if it is a contiguous slice of a longer phrase with
    a comparable author spread (the longer one is the 'real' meme).

    Only the top-`cap` phrases by author spread are de-overlapped (the rest are
    kept as-is) to keep this from blowing up on thousands of candidates. Uses an
    O(1) author-spread lookup so the pass is ~O(cap^2) cheap tuple comparisons.
    """
    if phrases.empty:
        return phrases
    df = phrases.sort_values("distinct_authors", ascending=False).reset_index(drop=True)
    spread = dict(zip(df["_tuple"], df["distinct_authors"]))
    head = list(df["_tuple"].head(cap))
    longers = [t for t in head if len(t) >= 5]  # only 5-grams can contain a 4-gram
    drop: set[tuple] = set()
    for short in head:
        Ls = len(short)
        sa = spread[short]
        for longer in longers:
            if len(longer) <= Ls:
                continue
            if any(longer[k : k + Ls] == short for k in range(len(longer) - Ls + 1)):
                if spread[longer] >= 0.6 * sa:
                    drop.add(short)
                    break
    out = df[~df["_tuple"].isin(drop)].drop(columns=["_tuple"]).reset_index(drop=True)
    return out


def memetic_table(token_memes: pd.DataFrame, phrase_memes: pd.DataFrame) -> pd.DataFrame:
    """Combine and rank. Boilerplate filter: drop candidates one author
    dominates (top_author_share > 0.5) — those aren't transmitted, just repeated."""
    cols = [
        "kind", "meme", "distinct_authors", "occurrences",
        "top_author_share", "first_seen", "last_seen", "active_days",
    ]
    frames = [d[cols] for d in (token_memes, phrase_memes) if not d.empty]
    allm = pd.concat(frames, ignore_index=True)
    allm = allm[allm["top_author_share"] <= 0.5]
    return allm.sort_values("distinct_authors", ascending=False).reset_index(drop=True)
