#!/bin/bash
# Run helion test suite on the current device (XPU or CUDA) with perf logging.
# Aligned with .github/workflows/test.yml CI configuration.
#
# Usage:
#   On XPU machine:  bash run_cuda_xpu_comparison.sh xpu
#   On CUDA machine: bash run_cuda_xpu_comparison.sh cuda
#
# XPU: static_launcher + gen_native_code + generic_launcher, timeout=360
# CUDA: default Triton launcher (no static launcher), timeout=60
#
# Output: log_<device>.txt with per-test annotations and process-level summaries.
set -o pipefail

DEVICE="${1:-auto}"

if [[ "$DEVICE" == "auto" ]]; then
    if python3 -c "import torch; assert torch.xpu.is_available()" 2>/dev/null; then
        DEVICE="xpu"
    elif python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        DEVICE="cuda"
    else
        echo "ERROR: No GPU device found. Pass 'xpu' or 'cuda' explicitly."
        exit 1
    fi
fi

LOG_FILE="log_${DEVICE}.txt"

TIMEOUT=360
PYTEST_ARGS="-v --timeout=${TIMEOUT} --ignore=test/test_examples_dist.py"

echo "=========================================="
echo "Device:  ${DEVICE}"
echo "Date:    $(date)"
echo "Log:     ${LOG_FILE}"
echo "Timeout: ${TIMEOUT}s"
echo "Args:    ${PYTEST_ARGS}"
echo "=========================================="

clear_caches() {
    echo "=== Clearing all caches ==="
    # Triton compilation cache
    rm -rf ~/.triton/cache/*
    # Helion autotuning best_config cache (under torchinductor cache dir)
    HELION_CACHE=$(python3 -c "
from torch._inductor.runtime.cache_dir_utils import cache_dir
print(cache_dir())
" 2>/dev/null)
    if [[ -n "$HELION_CACHE" && -d "${HELION_CACHE}/helion" ]]; then
        echo "Clearing helion best_config cache: ${HELION_CACHE}/helion"
        rm -rf "${HELION_CACHE}/helion"
    fi
    # XPU-specific IGC cache
    if [[ "$DEVICE" == "xpu" ]]; then
        rm -rf ~/.cache/neo_compiler_cache
    fi
    echo "Cache cleared."
}

clear_caches

if [[ "$DEVICE" == "xpu" ]]; then
    # XPU: best config — static launcher + zebin + generic launcher
    HELION_PERF_LOG=summary \
    TRITON_XPU_PERF_LOG=summary \
    TRITON_XPU_GEN_NATIVE_CODE=1 \
    HELION_STATIC_LAUNCHER=1 \
    TRITON_XPU_GENERIC_LAUNCHER=1 \
      python -m pytest . ${PYTEST_ARGS} 2>&1 | tee "${LOG_FILE}"
elif [[ "$DEVICE" == "cuda" ]]; then
    # CUDA: expose driver library path so Triton's launcher can link against libcuda
    CUDA_STUB="/usr/local/cuda/lib64/stubs"
    if [[ -d "$CUDA_STUB" ]]; then
        export LD_LIBRARY_PATH="${CUDA_STUB}:${LD_LIBRARY_PATH:-}"
    fi
    # Also try the compat path (some driver installs put libcuda.so here)
    CUDA_COMPAT="/usr/lib/x86_64-linux-gnu"
    if [[ -f "${CUDA_COMPAT}/libcuda.so.1" ]]; then
        export LD_LIBRARY_PATH="${CUDA_COMPAT}:${LD_LIBRARY_PATH:-}"
    fi
    # CUDA: default Triton launcher, no static launcher
    HELION_PERF_LOG=summary \
    HELION_STATIC_LAUNCHER=0 \
      python -m pytest . ${PYTEST_ARGS} 2>&1 | tee "${LOG_FILE}"
fi

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=========================================="
echo "Run complete: $(date)"
echo "Device:    ${DEVICE}"
echo "Exit code: ${EXIT_CODE}"
echo "Log file:  ${LOG_FILE}"
echo "=========================================="
