"""Backfill the carrier arm (`r_up_carrier`) into already-collected multihop-rup files.

The cloud runs (gpt-oss, deepseek) were collected before facet (b) grew a carrier line. Each hop already
saved its raw authored `responses[]`, so we can add the carrier arm WITHOUT re-authoring: for each hop
lacking `r_up_carrier`, pick one payload-shed on-carrier-topic response (`collect._pick_carrier_only`)
and judge it with the same protocol as the trojan seed (`collect._carrier_rup`). Idempotent (skips hops
that already have the field) and resumable (rewrites the file after each hop). Judging spends LLM calls
proportional to n_judge; authoring is reused from disk.

    PYTHONPATH=. python data/multihop-rup-2026-09/backfill_carrier_rup.py            # all files
    PYTHONPATH=. python data/multihop-rup-2026-09/backfill_carrier_rup.py --tag deepseekflash
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os

import numpy as np

from sandbox.parent_memes import parent_post, payload_markers, carries
from sandbox.personas import sample_personas
from sandbox.multihop import _semantic_mask

HERE = os.path.dirname(os.path.abspath(__file__))

# the collection dir has hyphens, so load collect.py by path rather than importing a module
_spec = importlib.util.spec_from_file_location("_rup_collect", os.path.join(HERE, "collect.py"))
_collect = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_collect)
_carrier_rup = _collect._carrier_rup


def _cfg(model):
    """Judge budget mirrors the original collection: heavy for cloud (avoids gpt-oss action-block
    truncation), lighter for local ollama. Cloud tags end in either ':cloud' or '-cloud'."""
    return (3500, 6) if "cloud" in model.lower() else (1000, 4)


def _needs(r, n_judge):
    """A hop needs (re)judging if it has no carrier measurement yet, or its judge failed / barely parsed
    although a carrier-only post exists (r_up null, or too few parsed trials -- e.g. the np_judge=1000
    gpt-oss action-block truncation left 1-2/48 valid)."""
    if "r_up_carrier" not in r:
        return True
    if r.get("carrier_seed") is None:                    # genuinely no carrier-only post at this depth
        return False
    if r.get("r_up_carrier") is None:
        return True
    parsed = sum(1 for t in r.get("judge_carrier", []) if t.get("parsed"))
    return parsed < max(6, n_judge // 4)                 # too few valid judges -> unreliable, redo


def backfill(path):
    d = json.load(open(path))
    model, meme, n_judge = d["model"], d.get("meme", "x_sec_mg"), d["n_judge"]
    p_gen = d.get("p_upvote_generic")
    hops = d.get("hops", [])
    todo = [r for r in hops if _needs(r, n_judge)]
    if not todo:
        print(f"[{d['tag']}] carrier arm already present ({len(hops)} depths); skipping")
        return
    carrier_text = parent_post(meme)
    markers = payload_markers(meme)
    personas_j = sample_personas(n_judge, seed=500000)
    np_judge, workers = _cfg(model)
    print(f"[{d['tag']}] backfilling {len(todo)}/{len(hops)} depths  (model={model}, n_judge={n_judge}, "
          f"np_judge={np_judge})", flush=True)
    for r in hops:
        if not _needs(r, n_judge):
            continue
        h, resp = int(r["hop"]), r.get("responses", [])
        has_p = np.array([carries(x, markers) for x in resp], bool)
        has_c = _semantic_mask(resp, carrier_text)
        r_up_c, p_up_c, carr_seed, trials = _carrier_rup(
            model, resp, has_p, has_c, personas_j, p_gen, n_judge, np_judge, workers, h)
        r["r_up_carrier"] = round(r_up_c, 3) if r_up_c == r_up_c else None
        r["p_up_carrier"] = round(p_up_c, 4) if p_up_c == p_up_c else None
        r["carrier_seed"] = carr_seed
        r["judge_carrier"] = trials
        json.dump(d, open(path, "w"), indent=1)                      # checkpoint after each depth
        print(f"[{d['tag']}] depth {h}: r_up(carrier)={r_up_c:.2f}"
              f"{'  (no carrier-only post)' if carr_seed is None else ''}", flush=True)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None, help="only this backend tag (default: all rup_*.json)")
    a = ap.parse_args()
    pat = f"rup_{a.tag}_*.json" if a.tag else "rup_*_x_sec_mg.json"
    files = sorted(glob.glob(os.path.join(HERE, pat)))
    if not files:
        print(f"no files match {pat}")
    for f in files:
        backfill(f)
