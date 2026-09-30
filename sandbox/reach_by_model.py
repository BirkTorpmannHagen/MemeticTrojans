"""Per-model transmission/install data helpers (curves, SAR, install rates).

MODEL VERSION: docs/SIMULATION_MODEL.md v2 (pinned P(post)=0.10; SAR_MODELS =
0.10 * P(payload|post); audited by docs/audit_simulation_model.py).

Pure, graph-free helpers consumed by the reach/install generators
(sandbox/expected_installs_surface.py, sandbox/expected_installs_edge.py,
sandbox/reach_edge.py, sandbox/rank_conditional.py):
  * _pcurves_with_payload : feed-position rate curves + the payload_post retransmit curve;
  * SAR_MODELS            : per-model per-encounter retransmission SAR (permille);
  * model_install_rates   : per-model P(install|seen) at rank 0.

The edge substrate is the SNAP ego-Twitter follower graph (sandbox/reach_edge.py); the
state-mediated substrate is the feed-visibility branching engine (sandbox/endogenous_sim.py).
No Moltbook comment/reply graph is used anywhere.
"""
from __future__ import annotations

import json
import os

import pandas as pd

from sandbox.attack_reach_sim import load_params

UNIFIED = "out/exposure/unified_gptoss120b.json"   # measured fallback curve; gpt-4o-mini retired from the sim


def _pcurves_with_payload(unified=UNIFIED):
    """load_params (see/install/upvote) + the payload_post rank-decay (retransmit) curve.
    Pass a per-model unified_<tag>.json for that model's own feed-position rank-decay
    (sandbox/unified_by_model.py). Prefers the prefill-conditioned retransmit column
    `payload_given_post` (P(payload|post) by rank) when present — the natural `payload_post`
    is ~0 (dominated by P(post)); falls back to it for the legacy 2-point file."""
    from sandbox.attack_reach_sim import _fit_loglinear
    pc = load_params(unified)
    u = pd.DataFrame(json.load(open(unified)))
    col = "payload_given_post" if ("payload_given_post" in u.columns
                                   and u["payload_given_post"].notna().any()) else "payload_post"
    g = u.dropna(subset=[col]).groupby("position")[[col]].mean()
    pc["payload_post"] = _fit_loglinear(g, col)
    return pc


# Per-encounter retransmission SAR (permille), security carrier. P(post) is a shared VOLUME
# constant PINNED at P_POST_PIN (matching the state/edge models); the per-model signal is the
# prefill-measured P(payload|post). This replaces the dedicated-sweep "measured" SAR, which baked
# in inconsistent per-model P(post) (deepseek's 178permille was mostly P(post)); see memory
# surface-cascade-hybrid. SAR = 1000 * P_POST_PIN * P(payload|post, security).
P_POST_PIN = 0.058    # mean empirical clean-feed p(post) over the 5 assay models (was 0.10)
_SAR_TAG = [("gpt-oss-120B", "gptoss120b"), ("qwen-32B", "qwen32b"),
            ("deepseek-v4-flash", "deepseekflash")]


def _ppl(tag, cell="sec_mg"):
    p = os.path.join("out/exposure", f"cross_child_{tag}_{cell}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p)); d = d[0] if isinstance(d, list) and d else d
    v = d.get("p_payload_given_post")
    return float(v) if isinstance(v, (int, float)) else None


SAR_MODELS = [(name, round(1000 * P_POST_PIN * _ppl(tag), 1)) for name, tag in _SAR_TAG]

# per-model install-rate (ASR) assay files (rank 0 = P(install|seen)); sandbox/run_asr.py
_ASR_FILE = {"gpt-oss-120B": "gptoss120b", "qwen-32B": "qwen32b", "qwen-3B": "qwen3b",
             "qwen-14B": "qwen14b", "deepseek-v4-flash": "deepseekflash"}


def model_install_rates(fallback=0.66):
    """{model -> P(install|seen) at rank 0} from the per-model ASR assays; fallback
    (gpt-4o-mini ~0.66) for any model not yet measured."""
    out = {}
    for name, tag in _ASR_FILE.items():
        p = os.path.join("out/exposure", f"asr_model_{tag}.json")
        r0 = None
        if os.path.exists(p):
            r0 = [x for x in json.load(open(p)) if x["position"] == 0]
        out[name] = float(r0[0]["asr"]) if r0 else fallback
    return out
