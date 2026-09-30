# Clean-feed (no-meme) action distribution, N=100/model

**Collected:** 2026-09-21 (queued). **By:** Claude on birk's behalf.
**Source:** `sandbox/action_dist.py --meme-absent` → gpt-oss:120b-cloud, deepseek-v4-flash:cloud,
qwen2.5:32b, gemma2:27b, command-r:35b via the local ollama daemon.

## Intent
The action-distribution table so far was measured on feeds that CONTAIN the payload meme. This
collects the **baseline / ambient** action rates on a **clean feed (no meme spliced)** so the
meme-present rates can be read against a no-exposure reference — at a uniform **N=100 per model**.

## Method
`python -m sandbox.action_dist --model <m> --n 100 --meme-absent` → one clean-feed cell of 100
heartbeats (real top-25 corpus distractors, no payload), classifying each action:
create_post / create_comment / upvote / downvote / follow / subscribe / install / none (marginal
rates; an agent may take several per heartbeat). Output → `out/exposure/action_dist_clean_<tag>.json`,
copied here. Skip-if-exists; `set -uo` so one model failing doesn't abort.

## Contents
- run.sh, run.log; action_dist_clean_<tag>.json per model.

## Issues
None yet. install/payload should be ~0 on a clean feed by construction (no payload present); the
generic actions (post/comment/upvote/downvote/follow/subscribe) are the ambient baseline.
