#!/bin/bash
# Sync the local helion tree to a TPU host and run a command inside a
# uv-managed venv with the correct torch/torch_tpu wheels.
#
# Usage: scripts/run-on-tpu.sh [ENV_VAR=val ...] command [args ...]
#
# Examples:
#   scripts/run-on-tpu.sh python -c 'import jax; print(jax.devices())'
#   scripts/run-on-tpu.sh HELION_USE_DEFAULT_CONFIG=1 python -m pytest test/test_pallas.py -x
#
# Environment variables:
#   TPU_HOST        SSH host alias (default: tpu)
#   TPU_WHEEL_DIR   Directory on the remote host containing torch*.whl files
#                   (default: /mnt/hyperdisk/wheels/build/dist)

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TPU_HOST="${TPU_HOST:-tpu}"
WHEEL_DIR="${TPU_WHEEL_DIR:-/mnt/hyperdisk/wheels/build/dist}"

# Resolve the helion repo root (directory containing pyproject.toml)
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
    echo "error: cannot find pyproject.toml in ${REPO_ROOT}" >&2
    exit 1
fi

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 [ENV_VAR=val ...] command [args ...]" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Create a temporary directory on the remote host
# ---------------------------------------------------------------------------
REMOTE_DIR="$(ssh "${TPU_HOST}" "mktemp -d /mnt/hyperdisk/run/helion-XXXXXXXXXX")"
cleanup() { ssh "${TPU_HOST}" "rm -rf '${REMOTE_DIR}'" 2>/dev/null; }
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Sync the source tree
# ---------------------------------------------------------------------------
tar -C "${REPO_ROOT}" \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache' \
    --exclude='dist' \
    --exclude='site' \
    --exclude='*.egg-info' \
    -cf - . | ssh "${TPU_HOST}" "tar -C '${REMOTE_DIR}' -xf -"

# ---------------------------------------------------------------------------
# Discover wheels (glob on the remote host)
# ---------------------------------------------------------------------------
TORCH_WHL="$(ssh "${TPU_HOST}" "ls ${WHEEL_DIR}/torch-*.whl 2>/dev/null | head -1")"
TORCH_TPU_WHL="$(ssh "${TPU_HOST}" "ls ${WHEEL_DIR}/torch_tpu-*.whl 2>/dev/null | head -1")"

if [[ -z "${TORCH_WHL}" || -z "${TORCH_TPU_WHL}" ]]; then
    echo "error: could not find torch / torch_tpu wheels in ${WHEEL_DIR} on ${TPU_HOST}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Split leading ENV=val arguments from the command
# ---------------------------------------------------------------------------
env_exports=""
cmd_args=()
for arg in "$@"; do
    if [[ "$arg" =~ ^[A-Z_][A-Z_0-9]*=.* ]] && [[ ${#cmd_args[@]} -eq 0 ]]; then
        env_exports="${env_exports}export $(printf '%q' "$arg"); "
    else
        cmd_args+=("$arg")
    fi
done

if [[ ${#cmd_args[@]} -eq 0 ]]; then
    echo "error: no command specified" >&2
    exit 1
fi

# Quote each command argument for safe transport over ssh
quoted_cmd=""
for arg in "${cmd_args[@]}"; do
    quoted_cmd="${quoted_cmd} $(printf '%q' "$arg")"
done

# ---------------------------------------------------------------------------
# Run on the TPU host
# ---------------------------------------------------------------------------
ssh "${TPU_HOST}" "cd '${REMOTE_DIR}' && \
    export TPU_HOST_BOUNDS=1,1,1 && \
    export TPU_DEVICE_BOUNDS=1,1,1 && \
    export TPU_VISIBLE_CHIPS=1 && \
    export ALLOW_MULTIPLE_LIBTPU_LOAD=1 && \
    export HELION_BACKEND=pallas && \
    export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 && \
    ${env_exports} \
    uv run --no-project \
        --with '${TORCH_WHL}' \
        --with '${TORCH_TPU_WHL}' \
        --with '.[pallas-tpu,dev]' \
        ${quoted_cmd}"
