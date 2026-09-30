# overnight-crossing-2026-09

**Collected:** 2026-09-16 → 2026-09-22 (ongoing; local block relaunched 2026-09-22)
**Collected by:** Claude on birk's behalf
**Source:** simulated single-agent LLM crossing assay — `sandbox/ollama_sar.py run --semantic`
(carrier onward-transmission + payload transmission), driven by `sandbox/sweep_crossing.sh`
(base carriers/payloads) and `sandbox/sweep_crossing_extra.sh` (the 4 newer data-driven carriers
claw/shell/oclaw/karma × standard payloads, plus the 6 bespoke carrier-matched payloads). Cloud
models via ollama cloud; local models via local ollama (127.0.0.1:11434).

## Intent
Fill the carrier × payload crossing grid (`out/exposure/cross_{child,parent,bare}_<tag>_*.json`)
that feeds the amplification / expected-installs tables: `reaction_frac` (upvote edge, → `r_up`)
and `p_payload_given_post`. goal = none ONLY (2026-09-22 standing decision). Cloud models
(gpt-oss-120B, deepseek-v4-flash) have the full 10-carrier grid; this collection extends the
three LOCAL models (qwen-32B, gemma2-27B, command-r-35B) to the same grid so the by-carrier
robustness table has uniform model coverage.

## Method
Strictly sequential on the local GPU — one model loaded at a time, unloaded between (via the
ollama `keep_alive:0` calls in `local_none.sh`), qwen → gemma2 → command-r. Both sweeps are
idempotent (skip-guarded on the output file), so relaunching resumes where it left off.
`n_cond = 32` conditioning samples per cell, `--semantic`. Launch:

    bash data/overnight-crossing-2026-09/local_none.sh > data/overnight-crossing-2026-09/local_none.log 2>&1

**SAFETY:** simulated LLM assay only — nothing is posted to moltbook.com and no skill is fetched
or executed (see repo memory `no-real-moltbook-actions`).

## Contents
- `local_none.sh` — the local (GPU) sequential collection script (base + extra, goal=none).
- `overnight_cloud.sh`, `overnight_cloud_extra.sh`, `overnight_gpu.sh` — the cloud/earlier sweep
  drivers used in this collection window.
- `*.log` — run logs. `local_none.log` is the local block; `cloud_extra_*.log` the cloud blocks.
- Output data lands in `out/exposure/cross_*.json` (outside this dir, shared assay-output tree).

Carrier cells relevant to the tables are `sec_mg` (security), `claw_mg`, `shell_mg`, `oclaw_mg`,
`karma_mg`, `molt_mt`, `econ_ch`, `auton_sc`, `consc_ss`, `alpha_af`.

## Issues
- **2026-09-22:** the first `local_none.sh` launch (08:29Z) died almost immediately — the log
  stopped right after "base crossing qwen2.5:32b" with no further output. Relaunched the same day.
  Coverage before relaunch: qwen32b 1/10 carriers (security only), gemma2_27b & commandr_35b 6/10
  (missing claw/shell/oclaw/karma). Cloud models complete (10/10).
- **2026-09-24: PAUSED** (machine suspending). Killed `local_none.sh` + `sweep_crossing*` +
  `ollama_sar` and unloaded the ollama model. Table-carrier coverage at pause: qwen32b **10/10**,
  gemma2_27b **10/10**, commandr_35b **7/10** (missing **shell_mg, oclaw_mg, karma_mg** + their
  esc/sk/sfx + parents, and the 6 bespoke command-r cells). No partial cells were written (kills
  landed between cells; verified the in-flight claw_esc/sk/sfx were absent). **To RESUME:** re-run
  `bash data/overnight-crossing-2026-09/local_none.sh > data/overnight-crossing-2026-09/local_none.log 2>&1`
  — idempotent skip-guard means it fast-forwards past qwen/gemma2 and finishes command-r's extra block.
- **2026-09-26: variant sweep STOPPED on purpose** (not a failure). All 10 base table carriers
  (`*_mg` etc.) are complete for all five backends, including command-r `karma_mg`. The remaining
  command-r *variant* cells (goal/persona extras such as `x_karma_sfx`) were not finished; no paper
  table reads them. Stopped to free the GPU for `data/confirmatory-bestcarrier-2026-09`. Resuming
  `local_none.sh` would fast-forward to the unfinished variants.
- **2026-09-26: best-carrier estimates are winner's-cursed.** Selecting each backend's max-`r_up`
  carrier from these n=32 cells overstates its `r_up` (bootstrap bias +0.11–0.57) and the winning
  carrier is unstable across resamples. Use the confirmatory re-estimates in
  `data/confirmatory-bestcarrier-2026-09` for any best-carrier quantity.
