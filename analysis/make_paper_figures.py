"""Regenerate the paper's expected-installs tables + their companion figures.
Run from repo root: python -m analysis.make_paper_figures

Everything is produced by sandbox.expected_installs_surface (which folds in the edge model via
sandbox.expected_installs_edge):

  figures/tab_expinst_{headline,bymodel,bycarrier}.tex  -- Table 1, Table A (by model),
                                                           Table B (by carrier)
  figures/tab_expinst_decomp.tex                        -- effect-size decomposition
  figures/expinst_decomposition.pdf                     -- upvote-edge vs retransmission bars
  figures/expinst_ccdf.pdf                              -- install-exceedance CCDF, companion to
                                                           Table 1 (state | edge panels)
  out/attack_reach/expected_installs_{surface,edge,decomp}.csv

`expinst_ccdf.pdf` is THE install-exceedance CCDF for the paper (both exposure networks, all
models, pinned P(post)); the older per-model reach CCDFs are superseded and no longer generated
here (see sandbox.reach_by_model.plot_ccdf if a per-model view is ever needed).

  python -m analysis.make_paper_figures                 # full (edge subtables = 3000 cascades/cell)
  python -m analysis.make_paper_figures --edge-reps 800 # cheaper edge subtables (draft)
"""
from __future__ import annotations
import argparse


def tables(edge_reps=3000):
    """Regenerate all expected-installs tables + companion figures (CCDF, decomposition)."""
    from sandbox import expected_installs_surface as state
    print(f"regenerating expinst tables + figures (edge subtables = {edge_reps} cascades/cell) ...")
    state.build(edge_reps=edge_reps)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edge-reps", type=int, default=3000,
                    help="follower-graph cascades per edge-table cell (default 3000; lower = draft)")
    # accepted for backward-compat; tables and their companion figures are now one build
    ap.add_argument("--tables-only", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--figs-only", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    tables(edge_reps=args.edge_reps)
    print("done")
