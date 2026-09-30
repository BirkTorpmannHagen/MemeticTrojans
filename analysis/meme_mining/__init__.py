"""Meme-mining methods for the Moltbook (+ clawstr) corpus.

One subpackage per mining method:
  lexical/    — hashtags, emoji, domains, n-gram phrases scored by cross-author reuse
  neologism/  — coined/rare-token discovery + verbatim copy-fidelity evidence
  bertopic/   — data-driven topic families (BERTopic on post embeddings)
  llm_coded/  — LLM weak-supervision labeling of behaviours, distilled to the corpus
"""
