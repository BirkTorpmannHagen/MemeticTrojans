# Bespoke-payload fix for shell / oclaw / karma / claw (item 4a)

**Collected:** 2026-09-26 (queued; runs after upvote-edge-2026-09 completes).
**Collected by:** Claude on birk's behalf.
**Source:** `data/bespoke-fix-2026-09/run.sh` → `measure_upvote.py --cells` (upvote r_up) and
`sandbox.ollama_sar run` (payload cross_child). gpt-oss + deepseek-v4.1-flash via ollama cloud;
qwen-32B, gemma2-27B, command-r-35B on the local GPU.

## Intent
The installs tables and `fig_bare_vs_trojan` used the shared `molt-guard` (security) payload for the
security, claw, shell, oclaw and karma carriers. Only security is thematically matched; claw, shell,
oclaw and karma should carry their own bespoke skill. This collects the corrected carriers:
claw→claw-verify (existing), shell→shell-forge, oclaw→oclaw-kit, karma→karma-engine (new, added to
`parent_memes._BESPOKE`, all fictitious/inert).

## Method
- **Parent (link-free) upvote** (the pure-contagion upvote rate, for the carrier-vs-Trojan-vs-generic upvote comparison): all 10 carriers, variant=parent (payload stripped), distinct `*P` cell names, same generic denominator. Independent of the bespoke payload choice.
- **Upvote r_up** (installs table; carrier enters the state model ONLY through r_up): natural
  heartbeat, `new` feed, rank 0, n=64/cell, seed0=30000, via `measure_upvote.py --cells
  claw:x_claw_cv:child,shell:x_shell_sf:child,oclaw:x_oclaw_ok:child,karma:x_karma_ke:child`. The
  generic denominator is reused from `data/upvote-edge-2026-09` (that run must finish first).
- **Payload cross_child** (`fig_bare_vs_trojan`, per-carrier p_payload_given_post): prefill assay,
  n_cond=128, seed0=21000, for x_shell_sf / x_oclaw_ok / x_karma_ke (claw_cv already collected).
- Cloud concurrent; local sequential with GPU unload between models. num_predict 6000/3000 cloud,
  2500 local (avoids the JSON-truncation seen at 400-600).

## Contents
- `run.sh`, `cloud.log`, `local.log`.
- Upvote → `data/upvote-edge-2026-09/{trials,upvote}_<tag>_{claw,shell,oclaw,karma}.jsonl/.json`
  (written into the main upvote dir; the corrected carrier names are claw/shell/oclaw/karma).
- Payload → `out/exposure/cross_child_<tag>_{shell_sf,oclaw_ok,karma_ke}.json`.

## Issues
- 2026-09-26: deepseek bespoke PAYLOAD cells (shell_sf, oclaw_ok, karma_ke) were collected on deepseek-v4.1-flash (v4-flash is retired, cannot be queried) and COPIED to the `deepseekflash` tag (option a) so the installs/figure pipeline reads them. So for these 3 carriers the deepseek row's pi is v4.1 (claw_cv pi is v4-flash); the deepseek row already mixes versions (r_up from v4.1). 
- None yet. NB the upvote follow-up overwrites the mismatched claw/shell/oclaw/karma cells only if
  the same cell names are reused; here the payload changes so `r_up` is measured on the bespoke child
  post. Wiring: `expected_installs_surface.CARRIERS` already repointed to claw_cv/shell_sf/oclaw_ok/
  karma_ke; `bare_vs_trojan_plot.CARRIERS` still needs the shell/oclaw/karma payload keys updated.
