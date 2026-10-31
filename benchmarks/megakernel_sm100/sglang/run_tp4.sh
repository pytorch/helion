#!/bin/bash
# One round of the SGLang DeepSeek-V3 NVFP4 MoE TP4 cold-L2 baseline (one benchmark process = four rank
# processes, each timing once).
#
#   GPUS4=<a,b,c,d> ROUND=<k> RESULTS=<dir> bash sglang/run_tp4.sh
#
# writes $RESULTS/sglang_deepseek_v3_moe_tp4_r<k>.json: rows deepseek_v3_moe_tp4/{spread,clustered,edge},
# median_us = the largest of the four ranks' medians, rank_medians_us = every rank's.  The run waits until
# no other process is on the four GPUs, and is redone if one appears during its timing.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../env.sh"
: "${GPUS4:?set GPUS4 to four idle B200s, e.g. 0,1,2,4}"
: "${ROUND:=1}"
: "${RESULTS:=$HERE/../results}"
: "${SGLANG_BENCH_SCRATCH:=$RESULTS/sglang_scratch}"
mkdir -p "$RESULTS"
wait_idle "$GPUS4"
sglang_python "$HERE/sglang_dsv3_moe_tp4.py" --gpus "$GPUS4" \
  --out "$RESULTS/sglang_deepseek_v3_moe_tp4_r${ROUND}.json" --scratch "$SGLANG_BENCH_SCRATCH/tp4_r${ROUND}"
