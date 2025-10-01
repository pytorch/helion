#!/usr/bin/env bash
set -euo pipefail

# Run the multistage tensor descriptor small-tile kernel across a range of block size configurations.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TEST_SCRIPT="${SCRIPT_DIR}/test_multistage_tensor_descriptor_alignment_small_tile.py"

if [[ ! -f "${TEST_SCRIPT}" ]]; then
    echo "Test script not found: ${TEST_SCRIPT}" >&2
    exit 1
fi

# TOTAL_NUMEL = 1024
declare -a BLOCK_SIZE_PAIRS_1024=(
    "4 256"
    "8 128"
    "16 64"
    "32 32"
    "64 16"
    "128 8"
    "256 4"
)

# TOTAL_NUMEL = 2048
declare -a BLOCK_SIZE_PAIRS_2048=(
    "4 512"
    "8 256"
    "16 128"
    "32 64"
    "64 32"
    "128 16"
    "256 8"
)

# TOTAL_NUMEL = 4096
declare -a BLOCK_SIZE_PAIRS_4096=(
    "4 1024"
    "8 512"
    "16 256"
    "32 128"
    "64 64"
    "128 32"
    "256 16"
)

# TOTAL_NUMEL = 8192
declare -a BLOCK_SIZE_PAIRS_8192=(
    "8 1024"
    "16 512"
    "32 256"
    "64 128"
    "128 64"
    "256 32"
    "512 16"
)

MACHINE=${MACHINE:-"unknown"}

for pairs in BLOCK_SIZE_PAIRS_1024 BLOCK_SIZE_PAIRS_2048 BLOCK_SIZE_PAIRS_4096 BLOCK_SIZE_PAIRS_8192; do
    declare -n pair_array="$pairs"
    echo "Testing ${pairs}:"
    for pair in "${pair_array[@]}"; do
        read -r block_bt block_v <<<"${pair}"
        if output=$(python "${TEST_SCRIPT}" --block-size-bt "${block_bt}" --block-size-v "${block_v}" 2>&1); then
            echo "  [${block_bt}, ${block_v}] -> pass on ${MACHINE}"
        else
            if grep -qi "out of resource" <<<"${output}"; then
                echo "  [${block_bt}, ${block_v}] -> OOM on ${MACHINE}"
            else
                echo "  [${block_bt}, ${block_v}] -> fail on ${MACHINE}"
            fi
        fi
    done
done
