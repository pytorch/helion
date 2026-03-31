#!/bin/bash
# Compare helion full test suite: baseline vs static launcher.
#
# Run 1: Baseline (static launcher disabled)
# Run 2: Static launcher enabled
#
# Both use generate_native_code=True (hardcoded in helion's default_launcher).
set -o pipefail

cd /home/stonepia/helion
source .venv/bin/activate

LOG_DIR="/home/stonepia/helion"
PYTEST_ARGS="-x --tb=short"

clear_caches() {
    echo "=== Clearing triton caches ==="
    rm -rf ~/.triton/cache/*
    rm -rf ~/.cache/neo_compiler_cache
    echo "Cache cleared."
}

echo "=========================================="
echo "Static Launcher Comparison"
echo "Date: $(date)"
echo "=========================================="

# ----------------------------------------
# RUN 1: Baseline (static launcher OFF)
# ----------------------------------------
echo ""
echo ">>> RUN 1/2: Baseline (static launcher disabled)"
echo ">>> Log: ${LOG_DIR}/log_baseline.txt"
echo ">>> Started: $(date)"
clear_caches

HELION_STATIC_LAUNCHER=0 \
  python -m pytest test/ ${PYTEST_ARGS} 2>&1 | tee "${LOG_DIR}/log_baseline.txt"

echo ">>> RUN 1 finished: $(date)"
echo ">>> Exit code: ${PIPESTATUS[0]}"

# ----------------------------------------
# RUN 2: Static launcher ON
# ----------------------------------------
echo ""
echo ">>> RUN 2/2: Static launcher enabled"
echo ">>> Log: ${LOG_DIR}/log_static.txt"
echo ">>> Started: $(date)"
clear_caches

HELION_STATIC_LAUNCHER=1 \
  python -m pytest test/ ${PYTEST_ARGS} 2>&1 | tee "${LOG_DIR}/log_static.txt"

echo ">>> RUN 2 finished: $(date)"
echo ">>> Exit code: ${PIPESTATUS[0]}"

# ----------------------------------------
# Summary
# ----------------------------------------
echo ""
echo "=========================================="
echo "All runs complete: $(date)"
echo ""
echo "Results:"
echo "  Baseline:  $(grep -E 'passed|failed' ${LOG_DIR}/log_baseline.txt | tail -1)"
echo "  Static:    $(grep -E 'passed|failed' ${LOG_DIR}/log_static.txt | tail -1)"
echo ""
echo "Log files:"
echo "  ${LOG_DIR}/log_baseline.txt"
echo "  ${LOG_DIR}/log_static.txt"
echo "=========================================="
