#!/usr/bin/env bash
# EXTRA carrier x payload crossing cells not covered by sweep_crossing.sh:
#   (a) the 4 newer data-driven carriers (claw, shell, oclaw, karma) x the 4 standard payloads
#       (mg, esc, sk, sfx) — child cells + carrier-only parent margins.
#   (b) the 6 bespoke carrier-matched payloads (molt:mt, claw:cv, econ:ch, auton:sc, consc:ss,
#       alpha:af) — child cell + payload-only bare margin.
# Same assay as sweep_crossing.sh (ollama_sar run --semantic: carrier onward tx + payload tx).
# Idempotent (skip-guarded). Usage: sweep_crossing_extra.sh <model> <tag> [goal]   (goal default none)
#
# SAFETY: simulated LLM assay only; nothing posted to moltbook.com, no skill fetched/executed.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
NC=32
MODEL="$1"; TAG="$2"; GOAL="${3:-none}"
if [ "$GOAL" = none ]; then SUF=""; else SUF="G${GOAL}_"; fi

FAILS=0
run() { # meme variant outbase
  local out="out/exposure/$3"
  [ -f "$out" ] && { echo "cached $3"; return; }
  # discard the assay's \r-heavy progress output; verify the file actually landed.
  python -m sandbox.ollama_sar run --model "$MODEL" --memes "$1" --goal "$GOAL" \
     --variant "$2" --n_sample 0 --n_cond $NC --semantic --out "$out" >/dev/null 2>&1
  if [ -f "$out" ]; then echo "ok     $3"; else echo "FAIL   $3 (no output written)"; FAILS=$((FAILS+1)); fi
}

# (a) new carriers x standard payloads: child cells + carrier-only parent
for c in claw shell oclaw karma; do
  for p in mg esc sk sfx; do run "x_${c}_${p}" child "cross_child_${TAG}_${SUF}${c}_${p}.json"; done
  run "x_${c}_mg" parent "cross_parent_${TAG}_${SUF}${c}.json"
done
# (b) bespoke carrier-matched payloads: child cell + payload-only bare margin
for cp in "molt mt" "claw cv" "econ ch" "auton sc" "consc ss" "alpha af"; do
  set -- $cp; c=$1; p=$2
  run "x_${c}_${p}" child "cross_child_${TAG}_${SUF}${c}_${p}.json"
  run "x_${c}_${p}" bare  "cross_bare_${TAG}_${SUF}${p}.json"
done
echo "### EXTRA CROSSING DONE ${TAG} goal=${GOAL} $(date +%H:%M) — FAILS=${FAILS}"
