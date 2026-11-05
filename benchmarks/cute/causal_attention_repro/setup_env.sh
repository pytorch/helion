#!/usr/bin/env bash
set -euo pipefail
CAUSAL_REPRO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${CAUSAL_REPRO_PYTHON:-}" ]]; then
    CAUSAL_REPRO_PYTHON="$(python3 "$CAUSAL_REPRO_ROOT/bootstrap_python.py")"
fi
python3 "$CAUSAL_REPRO_ROOT/setup.py" --python "$CAUSAL_REPRO_PYTHON"
