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
# Progress detection:
#   Each crash should block a different config (since blocked configs are
#   skipped on re-run). If the same config crashes twice, the autotuner
#   is stuck and we give up.
#
# Requirements:
#   - HELION_AUTOTUNE_CHECKPOINT_DIR must be set
#
# Usage:
#   HELION_AUTOTUNE_CHECKPOINT_DIR=/tmp/ckpt \
#       scripts/autotune_with_crash_recovery.sh -- COMMAND [ARGS...]
#
# Examples:
#   HELION_AUTOTUNE_CHECKPOINT_DIR=/tmp/autotune_ckpt \
#       scripts/autotune_with_crash_recovery.sh -- python train.py

set -uo pipefail

# --- Argument parsing ---
usage() {
    cat >&2 <<'EOF'
Usage: HELION_AUTOTUNE_CHECKPOINT_DIR=/path/to/dir \
           autotune_with_crash_recovery.sh -- COMMAND [ARGS...]
EOF
    exit "${1:-1}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
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

if [[ -z "${HELION_AUTOTUNE_CHECKPOINT_DIR:-}" ]]; then
    echo "Error: HELION_AUTOTUNE_CHECKPOINT_DIR must be set." >&2
    exit 1
fi

# --- Setup ---
checkpoint_dir="$HELION_AUTOTUNE_CHECKPOINT_DIR"
mkdir -p "$checkpoint_dir"

pending_file="$checkpoint_dir/_pending_config.txt"
bad_configs_file="$checkpoint_dir/_bad_configs.txt"

# --- Retry loop ---
attempt=0
last_config=""

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

        echo "[crash-recovery] Process crashed (exit code $exit_code, attempt $attempt)." >&2
        echo "[crash-recovery] Blocked config: $config" >&2

        # If the same config crashed again, the bad config is not being
        # skipped — the autotuner is stuck.
        if [[ "$config" == "$last_config" ]]; then
            echo "[crash-recovery] Same config crashed twice — the autotuner appears stuck." >&2
            echo "[crash-recovery] All bad configs have been recorded. You can re-run this script and it will resume from the latest checkpoint, skipping all previously recorded bad configs." >&2
            exit 1
        fi
        last_config="$config"

        echo "[crash-recovery] Restarting from checkpoint..." >&2
    else
        # No pending file — this is not a recoverable CUDA crash.
        # Propagate the original exit code.
        exit "$exit_code"
    fi
done
