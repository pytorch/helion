#!/bin/bash
# Wrapper for pyrefly check.
# On macOS, triton/cutlass are not installable, so we ignore their imports.
EXTRA=""
if [ "$(uname -s)" = "Darwin" ]; then
  for mod in triton "triton.*" cutlass "cutlass.*"; do
    EXTRA="$EXTRA --ignore-missing-imports $mod"
  done
fi
exec pyrefly check $EXTRA
