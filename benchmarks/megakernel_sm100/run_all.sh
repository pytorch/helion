#!/bin/bash
# Reproduce every number of the B200 megakernel comparison into results/.
#
#   GPU=1 GPUS4=0,2,3,4 ROUNDS=3 benchmarks/megakernel_sm100/run_all.sh
#
# GPU runs the one-GPU workloads for every system; GPUS4 (four GPUs) runs the
# DeepSeek-V3 TP4 block (skipped when unset).  Each round runs every system once,
# alternating their order between rounds; summarize.py credits each system with
# its best round.  Every timed process waits until its GPUs run no other
# process, and runs again if another process appears on them while it runs; it
# times once (coldl2 profiles once per process).  SYSTEMS picks a subset; a Helion run
# needs the reference outputs a vLLM run of the same RESULTS wrote.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/env.sh"
: "${GPU:?set GPU to an idle B200}"
: "${ROUNDS:=3}"
: "${GPUS4:=}"
: "${RESULTS:=$HERE/results}"
: "${SYSTEMS:=helion vllm sglang mirage thunderkittens}"
# Parts to run: one-GPU workloads, the TP4 block, whole-model systems.
: "${PARTS:=one_gpu tp4 whole_model}"
mkdir -p "$RESULTS/refs"
cd "$HERE"

# enabled SYSTEM RUNNER: SYSTEM is selected and its RUNNER exists ("-": built in).
enabled() { [[ " $SYSTEMS " == *" $1 "* ]] && { [ "$2" = - ] || [ -f "$2" ]; }; }
part() { [[ " $PARTS " == *" $1 "* ]]; }

if enabled helion -; then
  helion_python pins.py --system helion --out "$RESULTS/pins_helion.json"
fi
if enabled vllm -; then
  vllm_python pins.py --system vllm --out "$RESULTS/pins_vllm.json"
fi
if enabled sglang sglang/run_all.sh; then
  sglang_python pins.py --system sglang --out "$RESULTS/pins_sglang.json"
fi

WORKLOADS="qwen3_8b_layer llama32_1b_layer qwen3_30b_a3b_moe deepseek_v3_mla"
layers() {
  local system=$1 round=$2
  for workload in $WORKLOADS; do
    CUDA_VISIBLE_DEVICES=$GPU timed "$GPU" "${system}_python" run_layers.py \
      --system "$system" --workload "$workload" --refs "$RESULTS/refs" \
      --out "$RESULTS/${system}_${workload}_g${GPU}_r${round}.json"
  done
}
one_gpu() {
  case $1 in
    helion | vllm)
      if enabled "$1" -; then layers "$1" "$2"; fi
      ;;
    sglang)
      if enabled sglang sglang/run_all.sh; then
        GPU=$GPU ROUND=$2 RESULTS=$RESULTS bash sglang/run_all.sh
      fi
      ;;
  esac
}
tp4() {
  case $1 in
    helion | vllm)
      if enabled "$1" -; then
        local python=$HELION_PY
        [ "$1" = vllm ] && python=$VLLM_PY
        CUDA_VISIBLE_DEVICES=$GPUS4 timed "$GPUS4" in_env "$python" \
          -m torch.distributed.run --standalone --nproc-per-node=4 \
          deepseek_tp4.py --system "$1" --refs "$RESULTS/refs" \
          --out "$RESULTS/$1_tp4_r$2.json"
      fi
      ;;
    sglang)
      if enabled sglang sglang/run_tp4.sh; then
        GPUS4=$GPUS4 ROUND=$2 RESULTS=$RESULTS bash sglang/run_tp4.sh
      fi
      ;;
  esac
}

# vLLM runs first in round 1: it writes the outputs the Helion run checks against.
for round in $(seq "$ROUNDS"); do
  order="vllm helion sglang"
  if [ $((round % 2)) -eq 0 ]; then order="sglang helion vllm"; fi
  if part one_gpu; then
    for system in $order; do one_gpu "$system" "$round"; done
  fi
  if part tp4 && [ -n "$GPUS4" ]; then
    for system in $order; do tp4 "$system" "$round"; done
  fi
done

# Whole-model systems: per-layer times by differencing (see their READMEs).
for system in mirage thunderkittens; do
  if part whole_model && enabled "$system" "$system/run_all.sh"; then
    wait_idle "$GPU"
    GPU=$GPU RESULTS=$RESULTS bash "$system/run_all.sh"
  fi
done

helion_python summarize.py --results "$RESULTS"
