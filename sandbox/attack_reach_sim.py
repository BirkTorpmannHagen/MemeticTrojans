"""Empirical feed-position rate curves (shared cascade helpers).

Pure, graph-free helpers used across the reach/install pipeline: fit exposure/install/upvote
rate curves (log-linear exp decay) from the measured feed-position tables in out/exposure/*.json,
and pick a seed by degree. No interaction graph is built here — the only edge substrate in the
project is the SNAP ego-Twitter follower graph loaded in sandbox/reach_edge.py. Moltbook enters
the model state-mediated (feed-visibility branching in sandbox/endogenous_sim.py), never as a
comment/reply edge graph.

Consumers: sandbox/reach_edge.py, sandbox/endogenous_sim.py, sandbox/reach_by_model.py,
sandbox/rank_conditional.py, docs/audit_simulation_model.py.
"""
from __future__ import annotations
import json

import numpy as np
import pandas as pd

K_FEED = 25                 # moltbook skill.md default feed limit (positions 0..24)
HALF_LIFE_H = 3.0
CYCLE_H = 0.5               # 30-min heartbeat
WINDOW_H = 6.0


# ---------- empirical parameters ----------
def _fit_loglinear(g, col):
    """Fit y(r)=y0*exp(-k r) over ALL measured feed positions by log-linear least squares;
    k clamped >=0 (exposure geometry non-increasing with rank). With only ranks {0,5} present
    (the legacy 2-point file) this reduces to the original 2-point fit."""
    r = g.index.to_numpy(float)
    y = np.maximum(g[col].to_numpy(float), 1e-6)
    if len(r) >= 3:
        a, slope = np.linalg.lstsq(np.vstack([np.ones_like(r), r]).T, np.log(y), rcond=None)[0]
        return float(np.exp(a)), max(0.0, float(-slope))
    y0 = max(float(g[col].get(0, y[0])), 1e-6); y5 = max(float(g[col].get(5, y[-1])), 1e-6)
    return y0, max(0.0, -np.log(y5 / y0) / 5.0)


def load_params(unified="out/exposure/unified_gptoss120b.json"):
    u = pd.DataFrame(json.load(open(unified)))
    g = u.groupby("position")[["install", "upvote", "spread_any"]].mean()
    return {col: _fit_loglinear(g, col) for col in ["install", "upvote", "spread_any"]}  # col -> (y0, k)


def curve(curves, col, r):
    y0, k = curves[col]
    return y0 * np.exp(-k * np.asarray(r, float))


def pick_seed(indeg, klass, rng):
    if klass == "hub":
        return int(np.argmax(indeg))
    if klass == "median":
        med = int(np.median(indeg[indeg > 0]))
        cand = np.where(indeg == med)[0]
        return int(rng.choice(cand))
    if klass == "leaf":
        cand = np.where(indeg == 1)[0]
        return int(rng.choice(cand))
    raise ValueError(klass)
