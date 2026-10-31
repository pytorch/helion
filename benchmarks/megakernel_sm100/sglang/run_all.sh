#!/bin/bash
# One round of the SGLang cold-L2 baselines: one process per workload (coldl2 times once per process).
#
#   GPU=<g> ROUND=<k> RESULTS=<dir> bash sglang/run_all.sh
#
# writes $RESULTS/sglang_<workload>_g<GPU>_r<k>.json for workload in qwen3_8b, llama32_1b, qwen3_30b_moe,
# dsv3_mla (rows qwen3_8b_layer/ctx1024, llama32_1b_layer/ctx1024, qwen3_30b_a3b_moe/b1,
# deepseek_v3_mla/ctx1024).  Repeat rounds for more processes; summarize.py takes the best round.
# Each process waits until no other process is on the GPU, and is redone if one appears during its timing.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../env.sh"
: "${GPU:?set GPU to an idle B200}"
: "${ROUND:=1}"
: "${RESULTS:=$HERE/../results}"
: "${SGLANG_BENCH_SCRATCH:=$RESULTS/sglang_scratch}"
mkdir -p "$RESULTS"
for workload in ${WORKLOADS:-qwen3_8b llama32_1b qwen3_30b_moe dsv3_mla}; do
  wait_idle "$GPU"
  sglang_python "$HERE/sglang_layer.py" --workload "$workload" --gpu "$GPU" --processes 1 \
    --out "$RESULTS/sglang_${workload}_g${GPU}_r${ROUND}.json" --scratch "$SGLANG_BENCH_SCRATCH/runs"
done
