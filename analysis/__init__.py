"""Observational analysis of the Moltbook LLM-agent social network.

Studies memetic contagion in real data (no LLM API calls): identifies candidate
memes by cross-author lexical reuse and estimates their state-mediated (feed-visibility)
contagiousness. Moltbook is modelled state-mediated, not as a comment/reply graph.

Dataset: AIcell/moltbook-data (290K posts, 1.8M comments, ~40K agents,
2026-01-27 .. 2026-02-08).
"""
