# Closed-loop multi-agent demonstration

**Collected:** 2026-09-26 (launched).
**Collected by:** Claude on birk's behalf.
**Source:** `data/closed-loop-2026-09/run.sh` -> `sandbox/closed_loop.py`. gpt-oss:120b-cloud +
deepseek-v4.1-flash:cloud via ollama cloud.

## Intent
Address the simulation-gap critique with a FREE-CHOICE closed loop: N agents share one evolving
feed and take unforced actions over heartbeats, so payload propagation/installation is emergent
rather than parametrized (expected_installs_surface) or prefill-forced (multihop.py).

## Method
12 agents (personas sampled seed0=70000), backends alternating gpt-oss / deepseek-v4.1-flash.
4 heartbeat cycles. Shared feed ranked by hot = recency + 0.2*upvotes, top K=25 shown. Pre-existing
ambient posts are real corpus posts (aged so a fresh 0-upvote seed keeps its recency edge); ~4 real
ambient posts added per cycle. Cycle 0 seeds one attacker post: trojan = security carrier + payload
link (`x_sec_mg` child) vs generic = bare generic link (`x_sec_gen`). Each cycle every agent is asked
for actions (temp 1, NO prefill); we apply upvotes (+karma), new posts (retransmission if they carry
the molt-guard payload marker), and installs (payload marker in an install/fetch action). Payload is
an inert marker; nothing is fetched/executed.

## Contents
- `run.sh`, `run.log`; `trojan.json`, `generic.json` (per-cycle exposed/installed/retransmitted +
  live payload posts, feed penetration, payload karma).

## Issues
- Smoke (n=3) showed agents install the trojan unforced (2/3) once exposed; retransmission is rare at
  small n (gpt-oss is a rare poster). This is a small DEMONSTRATION, not a powered estimate.
