#!/usr/bin/env bash
# Validate p(retransmit)=p(retransmit|post)*p(post): direct natural sampling vs factored estimate.
# Cloud only (gpt-oss, deepseek-v4.1); 6-way concurrent; sequential across models (concurrency cap).
set -uo pipefail; cd "$(dirname "$0")/../.."; export PYTHONPATH=.
V="python data/retransmit-validation-2026-09/validate.py"
echo "### $(date -u +%FT%TZ) deepseek-v4.1 N=1000"
$V --model deepseek-v4.1-flash:cloud --tag deepseek41flash --n 1000 --n_cond 200 --np 800 --workers 6
echo "### $(date -u +%FT%TZ) gpt-oss N=2000"
$V --model gpt-oss:120b-cloud --tag gptoss120b --n 2000 --n_cond 200 --np 2000 --workers 6
echo "########## RETRANSMIT-VALIDATION DONE $(date -u +%FT%TZ) ##########"
