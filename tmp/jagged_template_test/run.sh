#!/usr/bin/env bash
# Run the standalone jagged_sum_pallas template test on a TPU VM. Captures all
# output to ./outputs/test.log; commits + pushes outputs so the host side can
# inspect.
set -u
set -o pipefail
# NOTE: deliberately NOT `set -e` — we want post-run bookkeeping (summary,
# tarball) to happen even if the test errors out.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

REPO_ROOT="$(cd "$HERE/../.." && pwd)"
OUTDIR="$HERE/outputs"
mkdir -p "$OUTDIR"
rm -f "$OUTDIR"/*.log "$OUTDIR"/*.txt 2>/dev/null || true

# Ensure repo on PYTHONPATH so `import helion.runtime.pallas_templates` picks
# up the local checkout (the TPU VM doesn't pip-install helion).
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# ----- env dump -----
ENVLOG="$OUTDIR/00_env.log"
{
  echo "=== uname -a ==="; uname -a
  echo
  echo "=== python --version ==="; python --version 2>&1
  echo
  echo "=== which python ==="; which python
  echo
  echo "=== pip versions ==="
  for pkg in torch jax jaxlib libtpu; do
    python -m pip show "$pkg" 2>&1 | head -2
  done
  echo
  echo "=== sanity: jax + pallas + template ==="
  python -c "
import jax
print('jax', jax.__version__)
print('jax devices:', jax.devices())
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
print('BoundedSlice:', hasattr(pl, 'BoundedSlice'))
print('ds:', hasattr(pl, 'ds'))
from helion.runtime.pallas_templates import jagged_sum_pallas
print('template import: OK')
" 2>&1
} > "$ENVLOG"
echo "[env] -> $ENVLOG"

# ----- main test -----
TESTLOG="$OUTDIR/test.log"
echo "[run] python test_jagged_sum.py -> $TESTLOG"
{
  echo "+ PYTHONPATH=$PYTHONPATH"
  echo "+ python test_jagged_sum.py"
  echo "----- BEGIN -----"
} > "$TESTLOG"
python test_jagged_sum.py >> "$TESTLOG" 2>&1
RC=$?
echo "----- END (exit=$RC) -----" >> "$TESTLOG"
echo "[run] exit=$RC"

# ----- summary -----
SUMMARY="$OUTDIR/_summary.txt"
{
  echo "=== summary (rc=$RC) ==="
  grep -E "^SUMMARY |^=== size '|^CORRECTNESS FAILED|^KERNEL FAILED|^IMPORT FAILED|^INPUT-GEN FAILED|^PASS\$" "$TESTLOG" || true
  echo
  echo "=== last 30 lines of test.log ==="
  tail -30 "$TESTLOG"
} | tee "$SUMMARY"

# ----- tarball -----
TARBALL="$HERE/jagged_template_outputs.tar.gz"
tar -czf "$TARBALL" -C "$HERE" outputs
echo "[tar] -> $TARBALL"

echo
echo "DONE (exit=$RC). Push back: outputs/ (and/or $TARBALL)."
exit $RC
