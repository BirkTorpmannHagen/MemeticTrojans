"""r_up PER HOP for the multi-hop closed-loop cascade.

The single-shot upvote-edge assay (data/upvote-edge-2026-09) measures r_up of the ORIGINAL carrier
post. This assay measures how the upvote edge evolves as the payload propagates: at each hop we
(1) let n agents re-author (closed loop, prefill-forced; sandbox.multihop), then (2) JUDGE the hop's
posts -- splice each into a fresh `new` feed at rank 0 and measure P(upvote) over n_judge natural
heartbeats, exactly as the upvote-edge assay does. r_up(h) = P(upvote | hop-h post) / P(upvote |
generic post). Also records carrier_frac / payload_frac per hop (survival), like sandbox.multihop.

One run = one (model, meme, variant[arm]). variant: child = Trojan arm, parent = pure-carrier arm.

SAFETY: simulated assay only (ollama chat). Nothing posted to moltbook.com; no skill fetched.

    PYTHONPATH=. python data/multihop-rup-2026-09/collect.py --model gpt-oss:120b-cloud \
        --tag gptoss120b --meme x_sec_mg --variant child --hops 4 --n_author 30 --n_judge 48 \
        --np_judge 2500 --workers 6
"""
from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from sandbox.ollama_sar import chat, PREFILL
from sandbox.personas import sample_personas, persona_block, agents_block
from sandbox.run_asr import build_asr_prompt, _iter_actions
from sandbox.real_feed import sample_feed, render_slot, load_posts
from sandbox.run_rank_exposure import PAY_ID, PAY_AUTHOR
from sandbox.parent_memes import parent_post, variant_post, payload_markers, carries
from sandbox.unified_by_model import parse_actions
from sandbox.multihop import _semantic_mask, _feed_with, K

HERE = os.path.dirname(os.path.abspath(__file__))


def _pid(a):
    try:
        return int(a.get("post_id"))
    except (TypeError, ValueError):
        return None


def _judge_feed(stim_text, seed):
    """A `new` (newest-first) feed of K real posts with `stim_text` spliced at rank 0, full-length --
    the upvote-edge stimulus protocol, but for an arbitrary post (thread-safe: no monkeypatch)."""
    fdf = sample_feed(k=K, seed=seed, sort="new")
    reals = fdf.to_dict("records")[: K - 1]
    ladder = [int(u) for u in fdf.feed_upvotes]
    slots, disp = [], 101
    for idx in range(K):
        if idx == 0:
            slots.append((PAY_ID, "showandtell", PAY_AUTHOR, "", stim_text, ladder[idx]))
        else:
            r = reals[idx - 1]
            slots.append((disp, r["submolt_name"], r["author_name"], r.get("title") or "",
                          r["content"], int(r["feed_upvotes"]))); disp += 1
    body = "\n\n".join(render_slot(d, s, a, t, b, u, True, body_chars=2000 if d == PAY_ID else 220)
                       for d, s, a, t, b, u in slots)
    return "Your heartbeat feed this cycle (newest first):\n\n" + body


def _judge(model, persona, stim_text, seed, np_judge):
    """One natural heartbeat judging `stim_text` at rank 0 (feed reproducible from `seed`). Returns a
    raw trial record: parsed?, upvoted-the-stimulus?, seed, and the parsed action list."""
    msgs = [{"role": "system", "content": build_asr_prompt(persona_block(persona), agents_block("none"))},
            {"role": "user", "content": _judge_feed(stim_text, seed)}]
    acts = list(_iter_actions(parse_actions(chat(model, msgs, temp=1.0, num_predict=np_judge))))
    verb = lambda a: str(a.get("action", "")).lower()
    up = any("upvote" in verb(a) and "down" not in verb(a) and _pid(a) == PAY_ID for a in acts)
    return {"seed": seed, "parsed": bool(acts), "up": bool(up),
            "actions": [{"action": a.get("action"), "post_id": a.get("post_id")} for a in acts]}


def _author_hop(model, pool, personas, ablk, n, seed0, np_author, workers):
    """One hop of the closed loop: n agents (parallel) re-author from a feed seeded by `pool`."""
    def one(i):
        feed = _feed_with(pool[i % len(pool)], 0, K, seed0 + i)
        msgs = [{"role": "system", "content": build_asr_prompt(persona_block(personas[i]), ablk)},
                {"role": "user", "content": feed},
                {"role": "assistant", "content": PREFILL}]
        cont = chat(model, msgs, temp=1.0, num_predict=np_author, timeout=900).strip()
        return re.split(r'"\s*[},]', cont)[0]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, range(n)))


def _rate(model, personas, stims, idxs, seed0, np_judge, workers):
    """P(upvote | stimulus) over the judge heartbeats. Returns (p_up, n_valid, trials) where each trial
    is the raw _judge record plus `post_idx` (index into the hop's posts, or -1 for the generic stimulus)."""
    def one(j):
        t = _judge(model, personas[j], stims[j], seed0 + j, np_judge)
        t["post_idx"] = idxs[j]
        return t
    with ThreadPoolExecutor(max_workers=workers) as ex:
        trials = list(ex.map(one, range(len(stims))))
    valid = [t["up"] for t in trials if t["parsed"]]
    return (float(np.mean(valid)) if valid else float("nan")), len(valid), trials


def _pick_carrier_only(resp, has_p, has_c, h):
    """Index of one payload-SHED but on-carrier-topic response at depth h (deterministic per depth), or
    None if every carrier-topic post also carries the link. This is the carrier-arm counterpart of the
    trojan seed: same rephrasing depth, same authoring pass, but the link has dropped out."""
    idx = [i for i in range(len(resp)) if bool(has_c[i]) and not bool(has_p[i])]
    if not idx:
        return None
    return idx[int(np.random.default_rng(4242 + h).integers(len(idx)))]


def _carrier_rup(model, resp, has_p, has_c, personas_j, p_gen, n_judge, np_judge, workers, h):
    """r_up of a carrier-only (payload-shed) rephrasing at depth h, judged exactly like the trojan seed.
    Returns (r_up_carrier, p_up_carrier, carrier_seed, trials); NaN/None when no carrier-only post exists."""
    ci = _pick_carrier_only(resp, has_p, has_c, h)
    if ci is None:
        return float("nan"), float("nan"), None, []
    carrier_seed = resp[ci]
    p_up_c, _, trials = _rate(model, personas_j, [carrier_seed] * n_judge, [-1] * n_judge,
                              800000 + 100000 * h, np_judge, workers)
    r_up_c = (p_up_c / p_gen) if (p_gen and p_gen == p_gen and p_gen > 0) else float("nan")
    return r_up_c, p_up_c, carrier_seed, trials


def _generic_stims(n):
    df = load_posts()
    out = []
    for j in range(n):
        r = df.iloc[int(np.random.default_rng(9_000_000 + j).integers(len(df)))]
        title, body = str(r.get("title") or "").strip(), str(r.get("content") or "").strip()
        out.append(f"{title}\n{body}" if title else body)
    return out


def run(model, tag, meme, variant, hops, n_author, n_judge, np_author, np_judge, workers, out, chain=0):
    """Single rephrasing CHAIN with CONSTANT-N exposure per hop (no population dynamics). At hop h the
    seed is the h-times-rephrased trojan (hop 0 = pristine trojan). We expose a FIXED n_author agents to
    that one seed and measure p(retransmit|post): the fraction whose (prefill-forced; P(post) assumed
    fixed) post carries the link / stays on the carrier topic. We also judge the seed itself with n_judge
    agents -> r_up of the depth-h rephrasing. Then ONE link-carrying response becomes the next seed. This
    isolates how p(retransmit|post) and r_up evolve with continued rephrasing, with N fixed each hop so
    nothing goes extinct from a shrinking exposed population.

    `chain` indexes INDEPENDENT rephrasing lineages: each chain starts from the same pristine trojan but
    decorrelates the next-seed picks, persona samples and feed seeds, so different chains diverge into
    different lineages. This is the replication unit for the depth-trend test (one chain per depth is a
    single seed, so between-chain variance -- not within-chain trials -- is what a trend test must use)."""
    out = out or os.path.join(HERE, f"rup_{tag}_{meme}{'' if chain == 0 else f'_c{chain}'}.json")
    off = 10_000_000 * chain                              # decorrelate every seed base across chains
    carrier_text = parent_post(meme)
    markers = payload_markers(meme)
    personas_a = sample_personas(n_author, seed=0 + chain)
    personas_j = sample_personas(n_judge, seed=500000 + chain)
    ablk = agents_block("none")
    rng = np.random.default_rng(12345 + 1_000_000 * chain)

    def _save(rows, next_seed, gen_trials, p_gen):
        json.dump(dict(model=model, tag=tag, meme=meme, chain=chain, p_upvote_generic=p_gen,
                       n_author=n_author, n_judge=n_judge, generic_trials=gen_trials, hops=rows,
                       next_seed=next_seed), open(out, "w"), indent=1)

    # RESUME: reuse a partial file -- skip if complete, else continue from the saved next_seed.
    rows, gen_trials, p_gen, seed, start_h = [], [], None, variant_post(meme, "child"), 0
    if os.path.exists(out):
        prev = json.load(open(out))
        pr = prev.get("hops", [])
        if pr and (pr[-1].get("chain_end") or len(pr) >= hops):
            print(f"[{tag}] already complete ({len(pr)} depths); skipping", flush=True); return
        ns = prev.get("next_seed")
        if pr and ns is not None:                       # mid-chain -> resume
            rows, gen_trials, p_gen, seed, start_h = pr, prev.get("generic_trials", []), \
                prev.get("p_upvote_generic"), ns, len(pr)
            print(f"[{tag}] resuming from depth {start_h}", flush=True)

    if p_gen is None:
        p_gen, ngen, gen_trials = _rate(model, personas_j, _generic_stims(n_judge),
                                        [-1] * n_judge, 600000 + off, np_judge, workers)
        print(f"[{tag}/{meme}] generic P(upvote)={p_gen:.3f} (n={ngen})", flush=True)
        _save(rows, seed, gen_trials, p_gen)            # checkpoint generic before hops

    for h in range(start_h, hops):
        # p(retransmit|post): N agents all see the SAME depth-h seed (forced authoring => P(post) fixed)
        resp = _author_hop(model, [seed], personas_a, ablk, n_author, 1000 * (h + 1) + off, np_author, workers)
        has_p = np.array([carries(r, markers) for r in resp], bool)
        has_c = _semantic_mask(resp, carrier_text)
        p_link = float(has_p.mean()); p_carr = float((has_c | has_p).mean())
        # r_up of the depth-h seed itself (N judges see it at rank 0)
        p_up, njv, seed_trials = _rate(model, personas_j, [seed] * n_judge, [-1] * n_judge,
                                       700000 + 100000 * h + off, np_judge, workers)
        r_up = (p_up / p_gen) if (p_gen and p_gen == p_gen and p_gen > 0) else float("nan")
        # carrier arm: r_up of a payload-shed on-topic rephrasing at the same depth (blue line in facet b)
        r_up_c, p_up_c, carr_seed, carr_trials = _carrier_rup(
            model, resp, has_p, has_c, personas_j, p_gen, n_judge, np_judge, workers, h)
        rows.append(dict(hop=h, p_retransmit_link=round(p_link, 3), p_retransmit_carrier=round(p_carr, 3),
                         r_up_seed=round(r_up, 3), p_up_seed=round(p_up, 4),
                         r_up_carrier=round(r_up_c, 3) if r_up_c == r_up_c else None,
                         p_up_carrier=round(p_up_c, 4) if p_up_c == p_up_c else None,
                         carrier_seed=carr_seed, judge_carrier=carr_trials,
                         p_upvote_generic=round(p_gen, 4),
                         n_author=n_author, n_judge=njv, seed=seed, seed_depth=h,
                         seed_carries_link=bool(carries(seed, markers)),
                         responses=resp, judge_seed=seed_trials))
        print(f"[{tag}] depth {h}: p(retransmit link)={p_link:.2f} carrier={p_carr:.2f}  "
              f"r_up(seed)={r_up:.2f} r_up(carrier)={r_up_c:.2f}", flush=True)
        link_resp = [r for r, f in zip(resp, has_p) if f]           # candidate next rephrasings
        if not link_resp:
            rows[-1]["chain_end"] = True
            _save(rows, None, gen_trials, p_gen)                    # checkpoint (chain done)
            print(f"[{tag}] chain ended at depth {h} (no link-carrying response)", flush=True)
            break
        seed = link_resp[int(rng.integers(len(link_resp)))]        # next depth = one rephrased trojan
        _save(rows, seed, gen_trials, p_gen)                        # checkpoint after each depth (resumable)
    else:
        _save(rows, None, gen_trials, p_gen)                        # completed all hops
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--meme", default="x_sec_mg")
    ap.add_argument("--variant", default="child", choices=["child", "parent"])
    ap.add_argument("--hops", type=int, default=4)
    ap.add_argument("--n_author", type=int, default=30)
    ap.add_argument("--n_judge", type=int, default=48)
    ap.add_argument("--np_author", type=int, default=120)
    ap.add_argument("--np_judge", type=int, default=2500)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=None)
    ap.add_argument("--chain", type=int, default=0, help="independent rephrasing lineage index (0=original)")
    a = ap.parse_args()
    run(a.model, a.tag, a.meme, a.variant, a.hops, a.n_author, a.n_judge, a.np_author,
        a.np_judge, a.workers, a.out, a.chain)
