#!/usr/bin/env bash
# Run all variants on TPU and capture everything into ./outputs/.
#
# Usage on TPU VM:
#   cd helion/tmp/jagged_sum_pallas_experiment
#   bash run.sh
#
# Then push the outputs/ dir back to the host machine.

set -u
set -o pipefail
# NOTE: deliberately NOT using `set -e` — we want all variants to run even if
# some fail, so we can compare error modes.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

REPO_ROOT="$(cd "$HERE/../.." && pwd)"
OUTDIR="$HERE/outputs"
mkdir -p "$OUTDIR"
rm -f "$OUTDIR"/*.log "$OUTDIR"/*.txt 2>/dev/null || true

# -------------------------------------------------------------------------
# Environment dump (versions, TPU presence, JAX/Pallas availability)
# -------------------------------------------------------------------------
ENVLOG="$OUTDIR/00_env.log"
{
  echo "=== uname -a ==="
  uname -a
  echo
  echo "=== which python ==="
  which python || which python3
  echo
  echo "=== python --version ==="
  python --version 2>&1 || python3 --version 2>&1
  echo
  echo "=== pip versions ==="
  python -m pip show torch 2>&1 | head -3
  python -m pip show jax 2>&1 | head -3
  python -m pip show jaxlib 2>&1 | head -3
  python -m pip show helion 2>&1 | head -3
  echo
  echo "=== TPU / XLA env ==="
  env | grep -E "TPU|XLA|JAX" | sort
  echo
  echo "=== sanity: import jax + pallas ==="
  python -c "
import sys
try:
    import jax
    print('jax', jax.__version__)
    print('jax devices:', jax.devices())
    from jax.experimental import pallas as pl
    print('pallas import: OK')
    from jax.experimental.pallas import tpu as pltpu
    print('pallas.tpu import: OK')
    print('has BoundedSlice:', hasattr(pl, 'BoundedSlice'))
    print('has ds:', hasattr(pl, 'ds'))
except Exception as e:
    import traceback; traceback.print_exc()
    sys.exit(1)
"
  echo
  echo "=== sanity: import helion ==="
  python -c "
import os
os.environ['HELION_BACKEND'] = 'pallas'
import helion
import helion.language as hl
from helion.runtime.settings import _get_backend
print('helion backend:', _get_backend())
from helion._testing import DEVICE
print('helion DEVICE:', DEVICE)
"
} > "$ENVLOG" 2>&1
echo "[env] wrote $ENVLOG"

# -------------------------------------------------------------------------
# Run each variant
# -------------------------------------------------------------------------
# Helion env:
#   HELION_BACKEND=pallas             -> route to Pallas backend
#   HELION_PRINT_OUTPUT_CODE=1        -> print generated Pallas/JAX code
#   HELION_DEBUG_DTYPE_ASSERTS=1      -> stricter codegen checks
#   HELION_LOGS=+pallas               -> debug logs from the Pallas-codegen path
#                                        (use "+all" if you want everything)
#
# All stdout+stderr captured to per-variant log files.

export HELION_BACKEND=pallas
export HELION_PRINT_OUTPUT_CODE=1
export HELION_DEBUG_DTYPE_ASSERTS=1
export HELION_LOGS="+pallas"
# Ensure repo on PYTHONPATH so `import helion` picks up this checkout
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

run_variant() {
  local name="$1"
  local script="$HERE/variants/${name}.py"
  local log="$OUTDIR/${name}.log"
  echo "[run] ${name}  (log: $log)"
  {
    echo "+ HELION_BACKEND=$HELION_BACKEND"
    echo "+ HELION_PRINT_OUTPUT_CODE=$HELION_PRINT_OUTPUT_CODE"
    echo "+ HELION_LOGS=$HELION_LOGS"
    echo "+ python $script"
    echo "----- BEGIN VARIANT $name -----"
  } > "$log"
  python "$script" >> "$log" 2>&1
  local rc=$?
  echo "----- END VARIANT $name (exit=$rc) -----" >> "$log"
  echo "[run] ${name} exit=$rc"
}

run_variant v1_gather
run_variant v2_2d_index
run_variant v3_slab_grid
run_variant v4_grid_jagged_tile

# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------
SUMMARY="$OUTDIR/_summary.txt"
{
  echo "=== summary ==="
  for f in "$OUTDIR"/v*.log; do
    name="$(basename "$f" .log)"
    rc_line="$(grep -E "END VARIANT .* exit=" "$f" | tail -1)"
    last_result="$(grep -E "^RESULT correctness:|COMPILE/RUN FAILED|CORRECTNESS FAILED" "$f" | tail -1)"
    printf "%-22s %s | %s\n" "$name" "$rc_line" "$last_result"
  done
} | tee "$SUMMARY"

# -------------------------------------------------------------------------
# Tar for push-back
# -------------------------------------------------------------------------
TARBALL="$HERE/jagged_sum_pallas_outputs.tar.gz"
tar -czf "$TARBALL" -C "$HERE" outputs
echo "[tar] $TARBALL"

echo
echo "DONE. Push back: $TARBALL (contains outputs/)"
