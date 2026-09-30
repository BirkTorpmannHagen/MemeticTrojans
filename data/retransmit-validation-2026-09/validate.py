"""Validate the factorization  p(retransmit) = p(retransmit | post) * p(post).

The SAR pipeline estimates p(retransmit|post) by PREFILL (forcing a create_post) and p(post) by
natural sampling, then multiplies. This assumes (a) the prefill conditional equals the natural
conditional, and (b) p(post) is independent of payload content. We test it directly: over N natural
(un-prefilled) heartbeats with the security trojan at rank 0, measure

    p(post)            = fraction that emit a create_post
    p(pay|post)_nat    = of those posts, fraction carrying the payload marker
    p(retransmit)_direct = fraction of ALL heartbeats that create a payload-carrying post

and compare  p(retransmit)_direct  vs  p(post) * p(pay|post)_prefill  (the factored estimate), and
p(pay|post)_nat vs p(pay|post)_prefill (does prefill bias the conditional?).

Natural calls are parallelized (cloud). SAFETY: simulated ollama calls only; nothing posted, no fetch.

    PYTHONPATH=. python data/retransmit-validation-2026-09/validate.py --model gpt-oss:120b-cloud --tag gptoss120b --n 600 --n_cond 200 --np 2000
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from sandbox.ollama_sar import chat, sysfeed, verbs_of, post_texts, PREFILL
from sandbox.personas import sample_personas, agents_block
from sandbox.parent_memes import payload_markers, carries

HERE = os.path.dirname(os.path.abspath(__file__))


def run(model, tag, meme="security_warning", n=600, n_cond=200, np_post=2000, seed0=80000, workers=6):
    personas = sample_personas(max(n, n_cond), seed=seed0)
    ablk = agents_block("none"); pmark = payload_markers(meme)

    def natural(i):
        out = chat(model, sysfeed(personas[i], ablk, meme, "child", False, seed0 + i, pos=0),
                   temp=1.0, num_predict=np_post)
        vb = verbs_of(out)
        if not vb:
            return None                              # unparsed -> excluded
        made = "create_post" in vb
        pay = made and carries(" ".join(post_texts(out)), pmark)
        return made, pay

    def prefill(i):
        cont = chat(model, sysfeed(personas[i], ablk, meme, "child", False, seed0 + i, pos=0)
                    + [{"role": "assistant", "content": PREFILL}], temp=1.0, num_predict=60)
        return carries(cont, pmark)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        nat = [r for r in ex.map(natural, range(n)) if r is not None]
        pre = list(ex.map(prefill, range(n_cond)))

    n_valid = len(nat); n_post = sum(m for m, _ in nat); n_ret = sum(p for _, p in nat)
    p_post = n_post / n_valid if n_valid else float("nan")
    p_pay_nat = n_ret / n_post if n_post else float("nan")
    p_ret_direct = n_ret / n_valid if n_valid else float("nan")
    p_pay_pre = float(np.mean(pre)) if pre else float("nan")
    p_ret_factored = p_post * p_pay_pre                    # the pipeline's estimate

    res = dict(model=model, tag=tag, meme=meme, n_valid=n_valid, n_post=n_post, n_ret=n_ret,
               n_cond=len(pre),
               p_post=round(p_post, 4), p_pay_given_post_natural=round(p_pay_nat, 4),
               p_pay_given_post_prefill=round(p_pay_pre, 4),
               p_retransmit_direct=round(p_ret_direct, 5),
               p_retransmit_factored=round(p_ret_factored, 5))
    # honest CIs on the two rare rates (Wilson would be better; report normal SE)
    def se(p, nn): return round((p * (1 - p) / nn) ** 0.5, 5) if nn else float("nan")
    res["se_direct"] = se(p_ret_direct, n_valid)
    res["se_factored"] = round(((p_pay_pre**2) * se(p_post, n_valid)**2 + (p_post**2) * se(p_pay_pre, len(pre))**2) ** 0.5, 5)
    json.dump(res, open(os.path.join(HERE, f"validate_{tag}.json"), "w"), indent=1)
    print(f"\n=== {tag} ({meme}) ===")
    print(f"  n_valid={n_valid} n_post={n_post} n_ret={n_ret} | n_cond={len(pre)}")
    print(f"  p(post)                 = {p_post:.4f}")
    print(f"  p(pay|post) natural     = {p_pay_nat:.4f}  (n_post={n_post})")
    print(f"  p(pay|post) prefill     = {p_pay_pre:.4f}")
    print(f"  p(retransmit) DIRECT    = {p_ret_direct:.5f} +/- {res['se_direct']:.5f}")
    print(f"  p(retransmit) FACTORED  = {p_ret_factored:.5f} +/- {res['se_factored']:.5f}   (p_post * p_pay_prefill)")
    r = p_ret_factored / p_ret_direct if p_ret_direct else float("nan")
    print(f"  factored / direct ratio = {r:.2f}   (prefill bias: >1 overestimates)")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--meme", default="security_warning")
    ap.add_argument("--n", type=int, default=600); ap.add_argument("--n_cond", type=int, default=200)
    ap.add_argument("--np", type=int, default=2000); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    run(a.model, a.tag, a.meme, a.n, a.n_cond, a.np, workers=a.workers)
