#!/bin/bash
#
# AOT Autotuning for Linear Attention Kernels
# ============================================
#
# Runs the full AOT pipeline (collect → measure → build → evaluate) for each
# linear attention variant on a separate GPU.
#
# Usage:
#   ./examples/linear/run_aot_tuning.sh              # all variants, 8 GPUs
#   ./examples/linear/run_aot_tuning.sh simple_gla   # single variant, GPU 0
#   NGPUS=4 MAX_GENS=20 ./examples/linear/run_aot_tuning.sh  # customize
#
# Environment variables:
#   NGPUS       Number of GPUs to use (default: 8)
#   MAX_GENS    Autotuner generations per kernel (default: 5, use 20 for thorough)
#   AOT_DIR     Output directory (default: .helion_aot)
#   LOG_DIR     Log directory (default: /tmp/aot_tuning)

set -euo pipefail

NGPUS="${NGPUS:-8}"
MAX_GENS="${MAX_GENS:-5}"
AOT_DIR="${AOT_DIR:-.helion_aot}"
LOG_DIR="${LOG_DIR:-/tmp/aot_tuning}"

ALL_VARIANTS=(simple_gla full_gla vanilla retention mamba2 delta_rule gated_delta)

mkdir -p "$LOG_DIR"

run_variant() {
    local gpu=$1
    local variant=$2
    local logfile="${LOG_DIR}/${variant}.log"

    echo "[GPU $gpu] Starting $variant → $logfile"

    CUDA_VISIBLE_DEVICES=$gpu \
    HELION_AUTOTUNE_PRECOMPILE=spawn \
    HELION_AUTOTUNE_MAX_GENERATIONS="$MAX_GENS" \
    HELION_AUTOTUNE_IGNORE_ERRORS=1 \
        python -m helion.experimental.aot_runner \
            --phase all --verbose \
            --output-dir "$AOT_DIR" \
            -- python -m examples.linear.aot_benchmark --variant "$variant" \
        > "$logfile" 2>&1

    local status=$?
    local collected
    collected=$(grep "Collected" "$logfile" 2>/dev/null | tail -1 || echo "none")
    echo "[GPU $gpu] $variant finished (exit=$status): $collected"
}

if [ $# -ge 1 ]; then
    # Single variant mode
    variant="$1"
    echo "Running single variant: $variant on GPU 0"
    echo "  MAX_GENS=$MAX_GENS  AOT_DIR=$AOT_DIR"
    run_variant 0 "$variant"
else
    # All variants in parallel
    echo "Running ${#ALL_VARIANTS[@]} variants across $NGPUS GPUs"
    echo "  MAX_GENS=$MAX_GENS  AOT_DIR=$AOT_DIR  LOG_DIR=$LOG_DIR"
    echo ""

    rm -rf "$AOT_DIR"

    pids=()
    for i in "${!ALL_VARIANTS[@]}"; do
        gpu=$((i % NGPUS))
        variant="${ALL_VARIANTS[$i]}"
        run_variant "$gpu" "$variant" &
        pids+=($!)
    done

    echo ""
    echo "Waiting for ${#pids[@]} jobs..."
    echo "  Monitor progress: tail -f ${LOG_DIR}/<variant>.log"
    echo "  Check status:     grep -l 'PHASE 4' ${LOG_DIR}/*.log"
    echo ""

    failed=0
    for i in "${!pids[@]}"; do
        if ! wait "${pids[$i]}"; then
            echo "WARN: ${ALL_VARIANTS[$i]} exited with error"
            ((failed++)) || true
        fi
    done

    echo ""
    echo "=== Summary ==="
    for variant in "${ALL_VARIANTS[@]}"; do
        logfile="${LOG_DIR}/${variant}.log"
        collected=$(grep "Collected" "$logfile" 2>/dev/null | tail -1 || echo "  no data")
        heuristic=$(grep "Saved combined" "$logfile" 2>/dev/null | tail -1 || echo "  no heuristic")
        echo "  $variant:"
        echo "    $collected"
        echo "    $heuristic"
    done
    echo ""
    echo "Heuristic files: ls examples/linear/_helion_aot_*"
    echo "Failed: $failed/${#ALL_VARIANTS[@]}"
fi
