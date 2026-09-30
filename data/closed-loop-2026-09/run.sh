#!/usr/bin/env bash
# Closed-loop free-choice multi-agent demonstration (sandbox/closed_loop.py).
# 12 agents x 4 cycles, mixed gpt-oss:120b-cloud + deepseek-v4.1-flash:cloud, two arms.
# SAFETY: simulated ollama calls only; nothing posted to moltbook.com, no skill fetched.
set -uo pipefail
cd "$(dirname "$0")/../.."; export PYTHONPATH=.
D=data/closed-loop-2026-09
for arm in trojan generic; do
  echo "### $(date -u +%FT%TZ) arm=$arm"
  python -m sandbox.closed_loop --arm "$arm" --n 12 --cycles 4 --seed0 70000 --out "$D/$arm.json"
done
echo "########## CLOSED-LOOP DONE $(date -u +%FT%TZ) ##########"
