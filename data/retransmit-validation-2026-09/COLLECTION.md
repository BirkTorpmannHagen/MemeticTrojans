# Retransmission factorization validation

**Collected:** 2026-09-28 (launched). **By:** Claude on birk's behalf.
**Source:** `data/retransmit-validation-2026-09/validate.py` (+ run.sh), ollama cloud.

## Intent
Test the load-bearing factorization p(retransmit)=p(retransmit|post)*p(post): the pipeline estimates
p(retransmit|post) by PREFILL and p(post) naturally, then multiplies. We check it against a DIRECT
natural-sampling estimate of p(retransmit), and compare the natural vs prefill conditional.

## Method
N natural (un-prefilled) heartbeats, security trojan at rank 0, count create_post and payload-carrying
create_post -> p(post), p(pay|post)_natural, p(retransmit)_direct. Plus n_cond prefill for
p(pay|post)_prefill. gpt-oss N=2000 (p(retransmit)~0.03, rare poster), deepseek-v4.1 N=1000. 6-way
concurrent. Compare p(retransmit)_direct vs p(post)*p(pay|post)_prefill.

## Contents
- validate.py, run.sh, run.log; validate_<tag>.json per model.

## Issues
- gpt-oss is a rare poster; even N=2000 gives ~60 retransmit events (SE~0.004) -- adequate for a
  consistency check, not a precise estimate. Not run on the 3 local models (GPU busy; cloud-only).
