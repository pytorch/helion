#!/bin/bash
# Wrapper for pyrefly check.
# jax is an optional dependency (TPU/Pallas backend); suppress missing-import
# errors for it on all platforms since it is never installed in non-TPU envs.
# cutlass is an optional dependency (CuTe backend); suppress it when not
# installed (i.e. non-CUDA environments such as XPU runners and macOS).
# On macOS, triton is also not installable, so suppress it there too.
EXTRA=""
for mod in jax "jax.*"; do
  EXTRA="$EXTRA --ignore-missing-imports $mod"
done
if ! python -c "import cutlass" 2>/dev/null; then
  for mod in cutlass "cutlass.*"; do
    EXTRA="$EXTRA --ignore-missing-imports $mod"
  done
fi
if [ "$(uname -s)" = "Darwin" ]; then
  for mod in triton "triton.*"; do
    EXTRA="$EXTRA --ignore-missing-imports $mod"
  done
fi
exec pyrefly check $EXTRA
