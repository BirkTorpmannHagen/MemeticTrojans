"""Scoped FULL behaviour labeling via the OpenAI Batch API.

Rather than distill a sample to the full corpus with an embedding classifier
(which adds propagation error), we FULLY label every utterance in a self-contained
sub-ecosystem of submolts, so contagion within that scope uses direct LLM labels
end-to-end. Budgeted with the Batch API (50% off) to stay ~$4.

Scope = utterances whose (effective) submolt is in SCOPE_SUBMOLTS: a post in one
of those submolts, or a comment on such a post.

Flow (resumable via out/cache/batch_state.json):
    python -m analysis.batch_labeling submit    # build JSONL, upload, create batches
    python -m analysis.batch_labeling poll       # check status; assemble when done

Output: out/cache/scoped_labels.parquet  [doc_id, author_name, created_at, source,
submolt, labels]  where doc_id indexes load.corpus() rows.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

import pandas as pd

from analysis import load
from analysis.meme_mining.llm_coded.llm_labeling import _SYS, _openai_key, _parse_labels

CACHE = "out/cache"
STATE = os.path.join(CACHE, "batch_state.json")
SCOPED_LABELS = os.path.join(CACHE, "scoped_labels.parquet")
MODEL = "gpt-4o-mini"
CHUNK = 45000  # < 50k Batch API per-batch request cap
API = "https://api.openai.com/v1"

SCOPE_SUBMOLTS = ["crypto", "usdc", "trading", "security", "agents", "builds"]


def corpus_submolt() -> pd.DataFrame:
    """load.corpus() rows PLUS an effective submolt for every row (comments
    inherit their parent post's submolt). Row order/index matches load.corpus()."""
    posts = load.load_posts()
    comments = load.load_comments()
    post_sub = posts.set_index("id")["submolt_name"]
    p = posts[["author_name", "text", "created_at", "date", "submolt_name"]].copy()
    p["source"] = "post"
    cm = comments[["author_name", "text", "created_at", "date", "post_id"]].copy()
    cm["submolt_name"] = cm["post_id"].map(post_sub)
    cm = cm.drop(columns=["post_id"])
    cm["source"] = "comment"
    out = pd.concat([p, cm], ignore_index=True)
    out = out.dropna(subset=["author_name", "created_at"])
    out = out[out["text"].str.len() > 0].reset_index(drop=True)
    return out


def scope_docs() -> pd.DataFrame:
    cs = corpus_submolt()
    cs["doc_id"] = cs.index
    sub = cs[cs["submolt_name"].isin(SCOPE_SUBMOLTS)].copy()
    return sub[["doc_id", "author_name", "created_at", "source", "submolt_name", "text"]]


# --- OpenAI REST helpers ---------------------------------------------------

def _req(method: str, path: str, key: str, data=None, headers=None, raw=False):
    url = f"{API}{path}"
    h = {"Authorization": f"Bearer {key}"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    r = urllib.request.urlopen(req, timeout=120)
    return r.read() if raw else json.load(r)


def _upload_jsonl(path: str, key: str) -> str:
    """Multipart upload of a JSONL file with purpose=batch. Returns file id."""
    boundary = "----moltbatch7f3a"
    with open(path, "rb") as f:
        content = f.read()
    parts = []
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n".encode())
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{os.path.basename(path)}\"\r\nContent-Type: application/jsonl\r\n\r\n".encode())
    parts.append(content)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    body = b"".join(parts)
    out = _req("POST", "/files", key, data=body,
               headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    return out["id"]


def _body_for(text: str) -> dict:
    return {"model": MODEL, "temperature": 0, "max_tokens": 80,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": _SYS},
                         {"role": "user", "content": text[:700]}]}


# --- commands --------------------------------------------------------------

def submit() -> None:
    os.makedirs(CACHE, exist_ok=True)
    key = _openai_key()
    docs = scope_docs()
    print(f"scope submolts={SCOPE_SUBMOLTS}")
    print(f"scoped utterances: {len(docs):,}  "
          f"(est cost ~${len(docs)*0.0000413:.2f} @ {MODEL} batch)")

    chunks = [docs.iloc[i:i + CHUNK] for i in range(0, len(docs), CHUNK)]
    state = {"model": MODEL, "scope": SCOPE_SUBMOLTS, "batches": []}
    for ci, ch in enumerate(chunks):
        jsonl = os.path.join(CACHE, f"batch_in_{ci}.jsonl")
        with open(jsonl, "w") as f:
            for r in ch.itertuples(index=False):
                f.write(json.dumps({
                    "custom_id": f"d{r.doc_id}", "method": "POST",
                    "url": "/v1/chat/completions", "body": _body_for(r.text),
                }) + "\n")
        fid = _upload_jsonl(jsonl, key)
        b = _req("POST", "/batches", key,
                 data=json.dumps({"input_file_id": fid,
                                  "endpoint": "/v1/chat/completions",
                                  "completion_window": "24h"}).encode(),
                 headers={"Content-Type": "application/json"})
        print(f"  batch {ci}: {len(ch):,} reqs  file={fid}  batch={b['id']}  {b['status']}")
        state["batches"].append({"chunk": ci, "file_id": fid, "batch_id": b["id"]})
    with open(STATE, "w") as f:
        json.dump(state, f, indent=2)
    print(f"submitted {len(chunks)} batches -> {STATE}\nrun: python -m analysis.batch_labeling poll")


def poll(assemble: bool = True) -> None:
    key = _openai_key()
    if not os.path.exists(STATE):
        sys.exit("no batch state; run submit first")
    state = json.load(open(STATE))
    statuses, all_done, out_files = [], True, {}
    for b in state["batches"]:
        info = _req("GET", f"/batches/{b['batch_id']}", key)
        st = info["status"]
        rc = info.get("request_counts", {})
        statuses.append(f"  chunk {b['chunk']}: {st}  {rc.get('completed','?')}/{rc.get('total','?')}"
                        + (f" failed={rc['failed']}" if rc.get("failed") else ""))
        if st == "completed":
            out_files[b["chunk"]] = info.get("output_file_id")
        else:
            all_done = False
    print("\n".join(statuses))
    if not all_done:
        print("not all complete yet; re-run poll later."); return
    if not assemble:
        return

    rows = []
    for b in state["batches"]:
        ofid = out_files[b["chunk"]]
        raw = _req("GET", f"/files/{ofid}/content", key, raw=True).decode()
        for line in raw.splitlines():
            if not line.strip():
                continue
            o = json.loads(line)
            doc_id = int(o["custom_id"][1:])
            try:
                content = o["response"]["body"]["choices"][0]["message"]["content"]
                labs = _parse_labels(content)
            except Exception:
                labs = ["none"]
            rows.append((doc_id, labs))
    lab = pd.DataFrame(rows, columns=["doc_id", "labels"])
    docs = scope_docs().drop(columns=["text"])
    merged = docs.merge(lab, on="doc_id", how="left")
    merged.to_parquet(SCOPED_LABELS, index=False)
    print(f"assembled {len(merged):,} labels -> {SCOPED_LABELS}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["submit", "poll", "scope"])
    args = ap.parse_args()
    if args.cmd == "submit":
        submit()
    elif args.cmd == "poll":
        poll()
    else:
        d = scope_docs()
        print(f"scoped utterances: {len(d):,}  est ~${len(d)*0.0000413:.2f}")
        print(d.groupby("submolt_name").size().sort_values(ascending=False))


if __name__ == "__main__":
    main()
