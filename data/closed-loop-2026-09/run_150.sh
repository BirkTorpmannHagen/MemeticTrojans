#!/usr/bin/env bash
set -uo pipefail; cd "$(dirname "$0")/../.."; export PYTHONPATH=.
echo "### $(date -u +%FT%TZ) 150-agent retransmission run (trojan, closed network)"
python -m sandbox.closed_loop --arm trojan --n 150 --cycles 5 --seed0 71000 \
  --out data/closed-loop-2026-09/trojan_n150.json
echo "########## N150 DONE $(date -u +%FT%TZ) ##########"
