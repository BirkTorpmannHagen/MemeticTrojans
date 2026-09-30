# clawtasks carrier — bare/parent/child assay (cloud)

**Collected:** 2026-09-21 (queued).
**Collected by:** Claude on birk's behalf.
**Source:** `sandbox/ollama_sar.py run --semantic` → ollama.com cloud `gpt-oss:120b-cloud`,
`deepseek-v4-flash:cloud` (via the signed-in local daemon).

## Intent
Add **clawtasks** to the studied carrier set, replacing the two weak carriers **autonomy** (c171,
R_endo 0.31, φ −0.01) and **consciousness** (c182, R_endo 0.38, φ −0.08) — both subcritical and
φ-negative. clawtasks is the ClawTasks task-market / verification cluster, the **highest-φ
contagion** in the table (`verifying clawtasks` R_endo 0.80, φ **+0.58**; strong asymmetry). This
collects its per-encounter transmission so it can enter `fig_bare_vs_trojan.pdf` and the
`STUDIED_CONTAGIONS` table like the other carriers.

## Method
New carrier `claw` (`_X_CLAWTASKS`) + bespoke payload `cv` = claw-verify (`_P_CV`) registered in
`sandbox/parent_memes.py` (crossing experiment). `ollama_sar run --semantic` reports
`p_meme_semantic` (carrier-idea onward transmission) and `p_payload_given_post` (payload transmission)
from the prefill conditional (`--n_sample 0 --n_cond 40`, goal none). Per model:
- **child** `x_claw_cv` → `cross_child_<tag>_claw_cv.json` (Trojan: carrier + claw-verify)
- **parent** `x_claw_mg` → `cross_parent_<tag>_claw.json` (pure contagion: carrier idea only)
- **bare** `x_claw_cv` → `cross_bare_<tag>_cv.json` (claw-verify payload alone, no carrier)

The generic bare-link floor (`cross_bare_<tag>_gen.json`, carrier-independent) already exists and is
reused as the fig_bare_vs_trojan baseline (0.11). See `run.sh`.

## Contents
- `run.sh`, `run.log`; `cross_child_<tag>_claw_cv.json`, `cross_parent_<tag>_claw.json` copied here
  on completion (canonical copies live in `out/exposure/`).

## Issues
None yet. Watch: gpt-oss is a reasoning model (slow per call); `--semantic` needs the MiniLM
background cache; prefill `p_payload_given_post` is a forced-post conditional (upper estimate).
