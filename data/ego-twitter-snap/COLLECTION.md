# SNAP ego-Twitter follower graph

**Collected:** 2026-09-20 (downloaded).
**Collected by:** Claude on birk's behalf.
**Source:** `https://snap.stanford.edu/data/twitter_combined.txt.gz` (SNAP ego-Twitter,
Leskovec & McAuley 2012). Saved to `data_ext/twitter_combined.txt.gz` (~10 MB; gitignored).

## Intent
An external social **follower graph** for the edge-mediated retransmission cascade
(`sandbox/reach_edge.py`). The Moltbook corpus has only a comment/interaction graph, not a
follower graph; ego-Twitter provides a realistic scale-free follower topology to study how
retransmission (SAR) compounds agent→follower→follower.

## Method
Direct download. Edge-list format: each line `"a b"` = account a **follows** b (whitespace-
separated integer node ids), so content authored by b appears in a's feed. Undirected? No — it is a
directed follow graph. Loaded by `sandbox.reach_edge.load_follower_graph`, which builds a follower
CSR (audience[v] = who sees v's posts = in-neighbours) plus per-node out-degree (feed breadth).

## Contents
- `data_ext/twitter_combined.txt.gz` (repo root data_ext/, gitignored) — the raw edge list.
- **81,306 nodes, 2,420,766 directed edges**; mean followers = mean followees = 29.8 (scale-free).

## Issues
None known so far. Caveats: it is a real Twitter follower graph, not Moltbook's own follow graph
(Moltbook exposes no follower graph), so it is a *structural stand-in* — the degree distribution and
scale-free hubs are realistic but the identities/communities are not Moltbook's. Node ids are the
original Twitter ids; `load_follower_graph` re-codes them to 0..N-1.
