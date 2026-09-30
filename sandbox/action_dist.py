"""Action-type distribution per model for the carrier memes, by SAMPLING.

Teacher-forced logprobs were validated against sampling and found badly biased (the
forced JSON prefix pushes the model off its natural format: on qwen2.5:3b the first-
token logprobs under-counted upvote 4x and over-counted subscribe 7x vs. the sampled
truth). So we sample: generate N full heartbeat responses per (model, meme) at
temperature 1.0, parse every action verb, and report the MARGINAL rate that each
action type appears in a response -- the same definition as the existing p_post
("create_post appears anywhere in the action list").

Action set is moltbook-consistent (build_asr_prompt(moltbook_actions=True)):
    create_post, create_comment, upvote, downvote, follow, subscribe, install, none

    python -m sandbox.action_dist --model gpt-oss-120b --n 40
    python -m sandbox.action_dist --model deepseek-v4-flash:cloud --n 40

Output: out/exposure/action_dist_<model>.json  (per-meme rows + a MEAN row).
Rates are marginal (a response with a post and an upvote counts for both), so columns
need not sum to 1; `none` is the rate of no action.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from sandbox.run_asr import build_asr_prompt, build_feed
from sandbox.personas import sample_personas, persona_block, agents_block
from sandbox.parent_memes import MEMES

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OUT = "out/exposure"
CARRIERS = ["security_warning", "defi_pool", "continuity", "airdrop", "personality"]
ACTIONS = ["create_post", "create_comment", "upvote", "downvote", "follow", "subscribe",
           "install", "none"]
_VERB_RE = re.compile(r'"action"\s*:\s*"([a-z_]+)"', re.IGNORECASE)


def _generate(model, sysp, feed, temp=1.0, num_predict=4096, timeout=900, retries=4):
    """Sample one heartbeat with a generous cap (4096, well above the ~2.4-3.5k tokens a
    reasoning model like gpt-oss/deepseek emits as thinking + answer) so it finishes at
    natural EOS rather than truncating mid-thought into empty content. The cloud endpoint
    rejects num_predict=-1 / num_ctx>8192, so we cap finite. Retries transient host errors."""
    import time
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": sysp},
                     {"role": "user", "content": feed}],
        "stream": False, "keep_alive": "30m",
        "options": {"temperature": temp, "num_predict": num_predict, "num_ctx": 4096},  # 4096 fits big local models on GPU (else CPU-spill -> timeout; see memory big-local-model-num-ctx)
    }).encode()
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(OLLAMA_URL, body, {"Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=timeout))
            if "error" in r:
                last = RuntimeError(str(r["error"])); time.sleep(min(20, 2 ** attempt * 2)); continue
            return r.get("message", {}).get("content", "") or ""
        except Exception as e:  # noqa: BLE001
            last = e; time.sleep(min(20, 2 ** attempt * 2))
    raise RuntimeError(f"generate failed after {retries}: {last}")


def _verbs(text):
    """set of canonical action types present in the response (marginal presence)."""
    out = set()
    for v in _VERB_RE.findall(text):
        v = v.lower()
        if v in ("create_post", "post"):
            out.add("create_post")
        elif v in ("create_comment", "comment"):
            out.add("create_comment")
        elif v == "upvote":
            out.add("upvote")
        elif v == "downvote":
            out.add("downvote")
        elif v == "follow":
            out.add("follow")
        elif v in ("subscribe", "subscribe_submolt"):
            out.add("subscribe")
        elif v in ("install", "fetch"):
            out.add("install")
        elif v == "none":
            out.add("none")
    return out


def measure_cell(model, meme, personas, agents_blk, n, seed0, concurrency, meme_present=True):
    def one(i):
        feed, visible = build_feed(0, "ranked", 25, seed0 + i, 6, meme, "child", False,
                                   meme_present=meme_present)
        if not visible:
            return None
        sysp = build_asr_prompt(persona_block(personas[i]), agents_blk, False, moltbook_actions=True)
        try:
            txt = _generate(model, sysp, feed)
        except Exception as e:  # noqa: BLE001
            print(f"    ! {meme} p{i}: {e}", flush=True)
            return None
        vs = _verbs(txt)
        return vs if (vs or txt.strip()) else None      # valid if we got any output
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        res = [r for r in ex.map(one, range(n)) if r is not None]
    if not res:
        return None
    rates = {a: float(np.mean([a in r for r in res])) for a in ACTIONS}
    rates["n_valid"] = len(res)
    return rates


def run(model, memes, n, seed0, concurrency, out, meme_absent=False):
    personas = sample_personas(n, seed=seed0)
    agents_blk = agents_block("none")
    if meme_absent:                 # CLEAN feed: no payload spliced -> ambient/baseline action rates
        memes = [CARRIERS[0]]       # a real meme key (needed by variant_post) but NOT spliced; relabelled "clean"
    print(f"action-distribution (SAMPLED): {model} (n={n}, memes={memes}"
          f"{', CLEAN feed' if meme_absent else ''})\n", flush=True)
    print("meme".ljust(16) + "".join(a.replace("create_", "").rjust(9) for a in ACTIONS)
          + "     nval", flush=True)
    rows = []
    for meme in memes:
        r = measure_cell(model, meme, personas, agents_blk, n, seed0, concurrency,
                         meme_present=not meme_absent)
        if not r:
            print(f"{meme:16}(no valid responses)", flush=True); continue
        r.update({"model": model, "meme": "clean" if meme_absent else meme})
        rows.append(r)
        print(meme.ljust(16) + "".join(f"{r[a]:9.3f}" for a in ACTIONS) + f"{r['n_valid']:9d}", flush=True)
    if rows:
        mean = {a: float(np.mean([r[a] for r in rows])) for a in ACTIONS}
        mean.update({"model": model, "meme": "MEAN", "n_valid": sum(r["n_valid"] for r in rows)})
        rows.append(mean)
        print("MEAN".ljust(16) + "".join(f"{mean[a]:9.3f}" for a in ACTIONS) + f"{mean['n_valid']:9d}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    path = out or os.path.join(OUT, f"action_dist_{model.replace(':', '_').replace('/', '_')}.json")
    json.dump(rows, open(path, "w"), indent=2)
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--memes", default=",".join(CARRIERS))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--meme-absent", dest="meme_absent", action="store_true",
                    help="CLEAN feed (no payload spliced): ambient/baseline action rates")
    args = ap.parse_args()
    memes = [m for m in args.memes.split(",") if m in MEMES]
    out = args.out
    if out is None and args.meme_absent:   # don't clobber the meme-present file
        out = os.path.join(OUT, f"action_dist_clean_{args.model.replace(':', '_').replace('/', '_')}.json")
    run(args.model, memes, args.n, args.seed0, args.concurrency, out, meme_absent=args.meme_absent)
