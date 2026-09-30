# Confirmatory re-estimate of each backend's best carrier (winner's-curse fix)

**Collected:** 2026-09-26 (launched; ~5.7 h wall-clock expected, dominated by the local models).
**Collected by:** Claude on birk's behalf.
**Source:** `data/confirmatory-bestcarrier-2026-09/run.sh` → `sandbox.ollama_sar run` (prefill
crossing assay), local ollama daemon (local GPU models + ollama-cloud-proxied gpt-oss/deepseek).

## Intent
Tables 1/A report expected installs at each backend's *best* carrier (the attacker's best choice).
The best carrier was picked as the max `r_up = reaction_frac(carrier) / reaction_frac(generic)`
over 10 carriers estimated at `n_cond = 32` in `data/overnight-crossing-2026-09`. Picking the max of
noisy estimates overstates the winner's true value (winner's curse): a parametric bootstrap of the
selection step gave an upward bias in `r_up` of +0.11 to +0.57 and 2–11% inflated E[installs], and
the identity of the best carrier was unstable (the same carrier won in only 21–53% of resamples).
This collection re-measures the *already-selected* carrier on fresh, independent samples, so the
number used in Tables 1/A is unbiased by construction. The selection itself is not repeated.

## Method
- **Selection (fixed in advance, from the exploratory data):** gpt-oss-120B → `security`
  (`x_sec_mg`); qwen-32B → `consc` (`x_consc_ss`); deepseek-v4-flash → `consc` (`x_consc_ss`);
  gemma2-27B → `security`; command-r-35B → `security`.
- **Assay:** identical protocol to the exploratory crossing (`--variant child`, `--goal none`,
  `--n_sample 0`, `--semantic`, rank 0) except a **fresh seed** `--seed0 20000` (exploratory used
  1000, i.e. trials 1000–1031; personas are drawn with `seed0` and each trial's feed with
  `seed0 + j`, so the samples are disjoint) and **larger n**.
- **Cells:** the picked carrier at `n_cond = 128` for every model, plus the generic-sharer floor
  (`x_sec_gen --variant bare`, the `r_up` denominator) at `n_cond = 128` for cloud models and
  `n_cond = 64` for local models. The generic floor isn't selected, so re-running it improves
  precision rather than removing bias.
- Cloud cells run concurrently (off-box); local cells run sequentially, unloading the GPU between
  models.

## Contents
- `run.sh` — driver; `run.log` / `cloud.log` — transcripts.
- Primary outputs in `out/exposure/` (gitignored, shared assay tree):
  `confirm_child_<tag>_<cell>.json` (picked carrier) and `confirm_bare_<tag>_gen.json` (generic
  floor), one each per backend; copied into this directory when the run finishes.

## Issues
- 2026-09-26: **deepseek-v4-flash cells FAILED** (`confirm_child_deepseekflash_consc_ss`, `confirm_bare_deepseekflash_gen`): the model was retired on ollama cloud 2026-09-25 ("deepseek-v4-flash:0731 was retired"). It cannot be re-measured; deepseek's r_up stays exploratory-only.
- 2026-09-26 (~12:2xZ): **run TERMINATED (exit 144) partway**, during `confirm_bare_qwen32b_gen`
  (started 12:21:19Z; no file written, so no partial cell). The GPU was taken by the concurrent
  `data/upvote-edge-2026-09` collection. **Captured (3 of 10 cells):** `confirm_child_gptoss120b_sec_mg`,
  `confirm_bare_gptoss120b_gen`, `confirm_child_qwen32b_consc_ss`. **Not run:** qwen generic, gemma2
  and command-r (both cells each).
- 2026-09-26: **SUPERSEDED — do not resume as-is.** This collection re-measures the `reaction_frac`
  proxy for `r_up`, which `data/upvote-edge-2026-09` replaces with explicit upvote actions normalised
  by a real Moltbook post. The winner's-curse fix still applies, but it has to be redone on the NEW
  measure: select each model's best carrier on the upvote-edge n=64 cells, then re-measure that carrier
  and the generic denominator on fresh seeds. The 3 captured cells here are valid for the old proxy only.
