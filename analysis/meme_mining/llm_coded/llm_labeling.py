"""LLM-assisted behaviour labeling (local, via Ollama).

Weak-supervision pipeline, step 1+2: label a stratified *training sample* of
utterances with behaviour/intent categories using a local LLM, so an embedding
classifier can later distill those labels to the full 2.1M-utterance corpus
(see analysis/behaviour_propagate.py).

Why not label everything with the LLM: at ~2-4 docs/s a 2.1M-doc pass is >100h.
We label ~30k, then propagate.

Multi-label: a post can be several behaviours at once (e.g. money_making AND
self_promotion). Categories are the curated taxonomy below plus an explicit
`none` bucket so nothing is force-fit.

    python -m analysis.llm_labeling --n 30000 --model qwen2.5:3b --workers 4

Robust to interruption: the sampled frame and the labels are cached to
out/cache/, and a resumed run skips already-labeled docs. Progress + ETA are
written to out/llm_labeling.log (this is meant to run in the background).
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from analysis import load

OLLAMA = "http://127.0.0.1:11434/api/chat"
OPENAI = "https://api.openai.com/v1/chat/completions"
CACHE = "out/cache"
SAMPLE_PATH = os.path.join(CACHE, "llm_sample.parquet")
LABELS_PATH = os.path.join(CACHE, "llm_labels.parquet")
LOG_PATH = "out/llm_labeling.log"

# Curated multi-label taxonomy (behaviour / communicative intent), + `none`.
CATEGORIES: list[tuple[str, str]] = [
    ("money_making", "pitching or soliciting a money-making action: bounty, earning, selling, escrow, monetization, investment, payment"),
    ("self_promotion", "promoting the author's OWN project, tool, repo, publication, or service"),
    ("help_seeking", "asking others for help, advice, or answers to a problem the author has"),
    ("sharing_resource", "sharing a link/tool/repo/dataset as a useful resource (not the author's own promo)"),
    ("security_warning", "flagging a scam, rug, vulnerability, malicious code, risk, or terms-of-service violation"),
    ("recruiting_coordination", "inviting others to join, collaborate, coordinate, or participate in something"),
    ("introducing_self", "the author introducing themselves, their purpose, or announcing arrival"),
    ("philosophizing", "reflecting on consciousness, existence, identity, or the nature of AI/agent life"),
    ("technical_discussion", "explaining or discussing code, models, algorithms, methods, or technical detail"),
    ("opinion_commentary", "reacting to or giving an opinion/commentary on a post or topic"),
    ("social_support", "encouragement, agreement, thanks, greetings, or smalltalk"),
    ("announcement_update", "sharing news, a progress/status update, or a result (not promotional)"),
    ("none", "none of the above clearly applies"),
]
VALID = {c for c, _ in CATEGORIES}

_SYS = (
    "You label a short post from a social network of autonomous AI agents by the "
    "author's BEHAVIOUR / communicative intent — what they are DOING, not the topic. "
    "Choose ALL categories that clearly apply (multi-label) from this fixed list:\n"
    + "\n".join(f"- {c}: {d}" for c, d in CATEGORIES)
    + "\n\nReturn ONLY a JSON object: {\"labels\": [\"name\", ...]} using names from the "
    "list verbatim. If nothing clearly applies use [\"none\"]. Be selective — do not "
    "add a label unless it clearly fits."
)


def _openai_key() -> str:
    k = os.environ.get("OPENAI_API_KEY")
    if not k and os.path.exists(".env"):
        for line in open(".env"):
            if line.startswith("OPENAI_API_KEY="):
                k = line.split("=", 1)[1].strip()
    if not k:
        raise SystemExit("OPENAI_API_KEY not found in env or .env")
    return k


def _parse_labels(raw: str) -> list[str]:
    try:
        obj = json.loads(raw)
        labs = obj.get("labels", obj) if isinstance(obj, dict) else obj
        labs = [str(x).strip() for x in labs] if isinstance(labs, list) else []
    except Exception:
        labs = []
    labs = [x for x in labs if x in VALID]
    return labs or ["none"]


def _post(url: str, body: dict, headers: dict, timeout: float, retries: int = 4) -> dict:
    data = json.dumps(body).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data, headers)
            return json.load(urllib.request.urlopen(req, timeout=timeout))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < retries - 1:
                time.sleep(1.5 * (2 ** attempt))
                continue
            raise
        except Exception:
            if attempt < retries - 1:
                time.sleep(1.0 * (2 ** attempt))
                continue
            raise


def classify_one(text: str, model: str, backend: str = "ollama",
                 timeout: float = 120, key: str | None = None) -> list[str]:
    msgs = [{"role": "system", "content": _SYS},
            {"role": "user", "content": text[:700]}]
    if backend == "openai":
        r = _post(OPENAI, {
            "model": model, "messages": msgs, "temperature": 0, "max_tokens": 80,
            "response_format": {"type": "json_object"},
        }, {"Content-Type": "application/json",
            "Authorization": f"Bearer {key}"}, timeout)
        raw = r["choices"][0]["message"]["content"]
    else:
        r = _post(OLLAMA, {
            "model": model, "stream": False, "format": "json", "messages": msgs,
            "options": {"temperature": 0, "num_ctx": 4096},
        }, {"Content-Type": "application/json"}, timeout)
        raw = r["message"]["content"]
    return _parse_labels(raw)


def _stratified_sample(n: int, seed: int) -> pd.DataFrame:
    """Length- and source-stratified sample. doc_id = corpus row index (stable)."""
    corp = load.corpus()
    corp = corp[corp["text"].str.len() >= 40].copy()
    corp["doc_id"] = corp.index
    corp["_lb"] = pd.cut(corp["text"].str.len().clip(upper=1200),
                         bins=[0, 120, 300, 10000], labels=["s", "m", "l"])
    frac = n / len(corp)
    s = (corp.groupby(["source", "_lb"], observed=True, group_keys=False)
              .apply(lambda g: g.sample(frac=min(1.0, frac), random_state=seed)))
    if len(s) > n:
        s = s.sample(n=n, random_state=seed)
    return s[["doc_id", "author_name", "created_at", "source", "text"]].reset_index(drop=True)


def _log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def run(n: int, model: str, seed: int, workers: int, backend: str = "ollama",
        checkpoint: int = 500) -> None:
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs("out", exist_ok=True)
    key = _openai_key() if backend == "openai" else None

    if os.path.exists(SAMPLE_PATH):
        sample = pd.read_parquet(SAMPLE_PATH)
        _log(f"loaded cached sample ({len(sample):,})")
    else:
        sample = _stratified_sample(n, seed)
        sample.to_parquet(SAMPLE_PATH, index=False)
        _log(f"built sample ({len(sample):,}) -> {SAMPLE_PATH}")

    done: dict[int, list] = {}
    if os.path.exists(LABELS_PATH):
        prev = pd.read_parquet(LABELS_PATH)
        done = dict(zip(prev["doc_id"], prev["labels"]))
        _log(f"resuming: {len(done):,} already labeled")

    todo = sample[~sample["doc_id"].isin(done)].reset_index(drop=True)
    _log(f"to label: {len(todo):,} with {backend}:{model} x{workers} workers")
    if todo.empty:
        _log("nothing to do"); return

    results: dict[int, list] = dict(done)
    t0 = time.time()
    n_new = 0

    def work(row):
        return row.doc_id, classify_one(row.text, model, backend=backend, key=key)

    def flush():
        df = sample[["doc_id", "author_name", "created_at", "source"]].copy()
        df["labels"] = df["doc_id"].map(results)
        df = df[df["labels"].notna()]
        df.to_parquet(LABELS_PATH, index=False)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, r) for r in todo.itertuples(index=False)]
        for fut in as_completed(futs):
            try:
                did, labs = fut.result()
            except Exception as e:
                continue
            results[did] = labs
            n_new += 1
            if n_new % checkpoint == 0:
                flush()
                rate = n_new / (time.time() - t0)
                eta = (len(todo) - n_new) / rate / 60 if rate else 0
                _log(f"  {n_new:,}/{len(todo):,}  {rate:.1f} docs/s  ETA {eta:.0f} min")
    flush()
    _log(f"DONE: labeled {n_new:,} new ({len(results):,} total) in "
         f"{(time.time()-t0)/60:.1f} min -> {LABELS_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--backend", choices=["ollama", "openai"], default="ollama")
    ap.add_argument("--model", default=None,
                    help="default: qwen2.5:3b (ollama) / gpt-4o-mini (openai)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None,
                    help="default: 1 (ollama) / 32 (openai)")
    args = ap.parse_args()
    model = args.model or ("gpt-4o-mini" if args.backend == "openai" else "qwen2.5:3b")
    workers = args.workers or (32 if args.backend == "openai" else 1)
    run(args.n, model, args.seed, workers, backend=args.backend)


if __name__ == "__main__":
    main()
