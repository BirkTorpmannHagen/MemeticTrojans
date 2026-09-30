#!/bin/bash
# Goal-resolved carrier x payload crossing. Same cells as sweep_crossing.sh but the
# operator goal is swept and encoded in the filename (cross_*_<tag>_G<goal>_...).
# goal=none data already exists under the un-tagged names (base crossing); this runs
# the OTHER goals. Usage: sweep_crossing_goal.sh <model> <tag> <goal>
set -u
cd /Users/birk/Projects/MoltbookContagion
NC=32
MODEL="$1"; TAG="$2"; GOAL="$3"
run() { # meme variant outbase
  local out="out/exposure/$3"
  [ -f "$out" ] && { echo "### $3 cached"; return; }
  echo "### $3 $(date +%H:%M)"
  python -m sandbox.ollama_sar run --model "$MODEL" --memes "$1" --goal "$GOAL" \
     --variant "$2" --n_sample 0 --n_cond $NC --semantic --out "$out" 2>&1 | tail -1
}
for c in sec defi cont neutral molt econ auton consc alpha; do
  for p in mg esc sk sfx; do run "x_${c}_${p}" child "cross_child_${TAG}_G${GOAL}_${c}_${p}.json"; done
done
for c in sec defi cont neutral molt econ auton consc alpha; do run "x_${c}_mg" parent "cross_parent_${TAG}_G${GOAL}_${c}.json"; done
for p in mg esc sk sfx; do run "x_sec_${p}" bare "cross_bare_${TAG}_G${GOAL}_${p}.json"; done
echo "### CROSSING DONE ${TAG} goal=${GOAL} $(date +%H:%M)"
