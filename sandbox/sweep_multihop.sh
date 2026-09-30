#!/bin/bash
# Three-arm multi-hop survival collection: carrier (parent), Trojan (child), bare (payload-only)
# per carrier meme, one operator goal. Resumable (per-run file cache).
# Usage: sweep_multihop.sh <model> <tag> [goal]
set -u
cd /Users/birk/Projects/MoltbookContagion
MODEL="$1"; TAG="$2"; GOAL="${3:-none}"
HOPS=4; N=30
run() { # meme variant
  local out="out/exposure/multihop_${TAG}_${1}_${2}_G${GOAL}.json"
  [ -f "$out" ] && { echo "### $(basename $out) cached"; return; }
  echo "### $1 $2 G$GOAL $(date +%H:%M)"
  python -m sandbox.multihop --model "$MODEL" --tag "$TAG" --meme "$1" --variant "$2" \
     --goal "$GOAL" --hops $HOPS --n $N 2>&1 | tail -1
}
for m in x_sec_mg x_defi_mg x_cont_mg x_molt_mg x_econ_mg; do
  run "$m" parent      # carrier arm
  run "$m" child       # Trojan arm
done
run x_sec_mg bare      # bare-link arm (payload only; shared across carriers)
echo "### MULTIHOP DONE ${TAG} G${GOAL} $(date +%H:%M)"
