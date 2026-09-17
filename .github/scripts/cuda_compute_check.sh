#!/usr/bin/env bash
set -euo pipefail

: "${HELION_CUDA_CHECK_ATTEMPTS:=3}"
: "${HELION_CUDA_CHECK_RETRY_DELAY_SECONDS:=20}"
: "${HELION_CUDA_CHECK_FORCE_FAIL_ATTEMPTS:=0}"

source .venv/bin/activate

run_cuda_compute_check_attempt() {
  local attempt="$1"

  if ((attempt <= HELION_CUDA_CHECK_FORCE_FAIL_ATTEMPTS)); then
    python - <<'PY'
import warnings

warnings.warn(
    "CUDA initialization: CUDA unknown error - this may be due to an incorrectly "
    "set up environment, e.g. changing env variable CUDA_VISIBLE_DEVICES after "
    "program start. Setting the available devices to be zero."
)
raise AssertionError("FATAL: CUDA not available")
PY
  else
    python - <<'PY'
import torch

assert torch.cuda.is_available(), "FATAL: CUDA not available"
n = torch.cuda.device_count()
assert n > 0, "FATAL: No CUDA devices found"
print(f"CUDA devices: {n}")

for i in range(n):
    dev = torch.device("cuda", i)
    a = torch.randn(256, 256, device=dev)
    (a @ a).sum().item()
    print(f"  Device {i} ({torch.cuda.get_device_name(i)}): OK")

print(f"All {n} devices healthy")
PY
  fi
}

for ((attempt = 1; attempt <= HELION_CUDA_CHECK_ATTEMPTS; attempt++)); do
  echo "::group::CUDA Compute Check attempt ${attempt}/${HELION_CUDA_CHECK_ATTEMPTS}"
  if run_cuda_compute_check_attempt "${attempt}"; then
    echo "::endgroup::"
    exit 0
  fi
  echo "::endgroup::"

  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi -L after failed attempt ${attempt}:"
    nvidia-smi -L || true
  fi

  if ((attempt < HELION_CUDA_CHECK_ATTEMPTS)); then
    echo "Retrying CUDA Compute Check in ${HELION_CUDA_CHECK_RETRY_DELAY_SECONDS}s"
    sleep "${HELION_CUDA_CHECK_RETRY_DELAY_SECONDS}"
  fi
done

echo "CUDA Compute Check failed after ${HELION_CUDA_CHECK_ATTEMPTS} attempts" >&2
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi
exit 1
