#!/usr/bin/env bash
# Autotuner crash recovery wrapper.
#
# Runs a command (typically a Python script that calls helion autotuning)
# in a retry loop. When the process crashes due to an unrecoverable CUDA
# error (illegal memory access, misaligned address, etc.), the autotuner
# leaves a "_pending_config.txt" breadcrumb in the checkpoint directory.
# This script detects that file, records the poison config in
# "_bad_configs.txt", and re-runs the command. On re-run the autotuner
# loads its checkpoint and skips the bad config.
#
# Requirements:
#   - The autotuner must have checkpointing enabled
#     (HELION_AUTOTUNE_CHECKPOINT_DIR or --checkpoint-dir)
#
# Usage:
#   scripts/autotune_with_crash_recovery.sh [OPTIONS] -- COMMAND [ARGS...]
#
# Options:
#   --checkpoint-dir DIR   Directory for checkpoint/recovery files
#                          (default: $HELION_AUTOTUNE_CHECKPOINT_DIR)
#   --max-attempts N       Maximum crash recovery attempts (default: 10)
#   -h, --help             Show this help message
#
# Examples:
#   scripts/autotune_with_crash_recovery.sh \
#       --checkpoint-dir /tmp/autotune_ckpt -- python train.py
#
#   HELION_AUTOTUNE_CHECKPOINT_DIR=/tmp/ckpt \
#       scripts/autotune_with_crash_recovery.sh -- python train.py

set -uo pipefail

# --- Defaults ---
checkpoint_dir="${HELION_AUTOTUNE_CHECKPOINT_DIR:-}"
max_attempts=10

# --- Argument parsing ---
usage() {
    cat >&2 <<'EOF'
Usage: autotune_with_crash_recovery.sh [OPTIONS] -- COMMAND [ARGS...]

Options:
  --checkpoint-dir DIR   Directory for checkpoint/recovery files
                         (default: $HELION_AUTOTUNE_CHECKPOINT_DIR)
  --max-attempts N       Maximum crash recovery attempts (default: 10)
  -h, --help             Show this help message
EOF
    exit "${1:-1}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint-dir)
            checkpoint_dir="$2"
            shift 2
            ;;
        --max-attempts)
            max_attempts="$2"
            shift 2
            ;;
        -h|--help)
            usage 0
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Error: unknown option '$1'" >&2
            usage 1
            ;;
    esac
done

if [[ $# -eq 0 ]]; then
    echo "Error: no command specified after --" >&2
    usage 1
fi

if [[ -z "$checkpoint_dir" ]]; then
    echo "Error: checkpoint directory required." >&2
    echo "Set --checkpoint-dir or HELION_AUTOTUNE_CHECKPOINT_DIR." >&2
    exit 1
fi

# --- Setup ---
mkdir -p "$checkpoint_dir"
export HELION_AUTOTUNE_CHECKPOINT_DIR="$checkpoint_dir"

pending_file="$checkpoint_dir/_pending_config.txt"
bad_configs_file="$checkpoint_dir/_bad_configs.txt"

# --- Retry loop ---
attempt=0
while true; do
    attempt=$((attempt + 1))

    # Run the user command (don't use set -e, capture exit code manually)
    "$@"
    exit_code=$?

    if [[ $exit_code -eq 0 ]]; then
        exit 0
    fi

    # Check if the autotuner left a pending config breadcrumb
    if [[ -f "$pending_file" ]]; then
        config=$(cat "$pending_file")
        rm -f "$pending_file"
        echo "$config" >> "$bad_configs_file"

        echo "[crash-recovery] Process crashed (exit code $exit_code, attempt $attempt/$max_attempts)." >&2
        echo "[crash-recovery] Blocked config: $config" >&2

        if [[ $attempt -ge $max_attempts ]]; then
            echo "[crash-recovery] Exceeded maximum recovery attempts ($max_attempts)." >&2
            exit 1
        fi

        echo "[crash-recovery] Restarting from checkpoint..." >&2
    else
        # No pending file — this is not a recoverable CUDA crash.
        # Propagate the original exit code.
        exit "$exit_code"
    fi
done
