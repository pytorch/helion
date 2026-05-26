#!/usr/bin/env bash
# Diagnose the 2.5% inaccuracy in examples/jagged_sum_tpu.py.
#
# Captures Helion's lowered Pallas code (HELION_PRINT_OUTPUT_CODE=1) and the
# per-seed correctness analysis to outputs/.

set -u
set -o pipefail
# Not `set -e` — want the summary + tarball even when the kernel fails.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

REPO_ROOT="$(cd "$HERE/../.." && pwd)"
OUTDIR="$HERE/outputs"
mkdir -p "$OUTDIR"
rm -f "$OUTDIR"/*.log "$OUTDIR"/*.txt 2>/dev/null || true

# Env
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HELION_BACKEND=pallas
export HELION_PRINT_OUTPUT_CODE=1
export HELION_AUTOTUNE_EFFORT=none       # fixed config -> diagnosable
export HELION_LOGS="+pallas"

LOG="$OUTDIR/diagnose.log"
{
  echo "+ HELION_BACKEND=$HELION_BACKEND"
  echo "+ HELION_PRINT_OUTPUT_CODE=$HELION_PRINT_OUTPUT_CODE"
  echo "+ HELION_AUTOTUNE_EFFORT=$HELION_AUTOTUNE_EFFORT"
  echo "+ HELION_LOGS=$HELION_LOGS"
  echo "+ PYTHONPATH=$PYTHONPATH"
  echo "+ python diagnose.py"
  echo "----- BEGIN -----"
} > "$LOG"
python diagnose.py >> "$LOG" 2>&1
RC=$?
echo "----- END (exit=$RC) -----" >> "$LOG"
echo "[run] python diagnose.py exit=$RC (log: $LOG)"

# Concise summary: per-seed pass/fail + the cross-seed comparison + last
# portion of the log (which contains tracebacks if anything failed).
SUMMARY="$OUTDIR/_summary.txt"
{
  echo "=== summary (rc=$RC) ==="
  grep -E "^RESULT |^=== variant=|^=== variant summary|^variant=|^  variant\[" "$LOG" || true
  echo
  echo "=== tail of diagnose.log ==="
  tail -60 "$LOG"
} | tee "$SUMMARY"

echo
echo "DONE (exit=$RC). Push outputs/ back."
exit $RC
