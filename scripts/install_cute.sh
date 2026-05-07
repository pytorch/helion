#!/usr/bin/env bash
# Install the pinned nvidia-cutlass-dsl release.
#
# The pin lives in .github/ci_commit_pins/nvidia_cutlass_dsl.txt. Helion
# integrates tightly with the cutlass DSL's internal APIs, so each upstream
# release tends to break Helion's cute backend (region isolation rules,
# kwarg renames, register API signature, etc.). Pinning a known-good
# PyPI version + a periodic upgrade-and-fix sweep is more sustainable than
# tracking the latest pre-release.
#
# A PyPI version is immutable once uploaded (the wheel bytes are locked,
# and ``requires_dist`` pins the libs-base / libs-cu13 wheels to the same
# version), but can be *yanked* — which would surface as an install
# failure, not a silent content change. For ironclad reproducibility you
# would clone https://github.com/NVIDIA/cutlass at a specific commit and
# build the Python DSL from source (~minutes of CUDA/C++ build per run).
# The PyPI pin is the practical tradeoff here.
#
# Usage:
#   ./scripts/install_cute.sh           # auto-detects cu12 vs cu13 from torch
#   ./scripts/install_cute.sh cu12
#   ./scripts/install_cute.sh cu13
#
# Outside CI (no `uv` on PATH) this falls back to plain `python -m pip`.

set -euo pipefail

# If unspecified, derive cu12/cu13 from the installed torch's CUDA build.
if [[ $# -ge 1 ]]; then
  VARIANT="$1"
else
  TORCH_CUDA="$(python -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null || echo "")"
  case "$TORCH_CUDA" in
    13.*|13)  VARIANT="cu13" ;;
    12.*|12)  VARIANT="cu12" ;;
    "")
      echo "error: torch is not installed or has no CUDA build; pass cu12 or cu13 explicitly" >&2
      exit 1
      ;;
    *)
      echo "error: torch CUDA version '$TORCH_CUDA' is not supported (expected 12.x or 13.x)" >&2
      echo "       pass cu12 or cu13 explicitly to override" >&2
      exit 1
      ;;
  esac
  echo "==> Detected torch CUDA $TORCH_CUDA -> variant $VARIANT"
fi

case "$VARIANT" in
  cu12|cu13) ;;
  *)
    echo "error: unsupported variant '$VARIANT' (expected cu12 or cu13)" >&2
    exit 1
    ;;
esac

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIN_FILE="$REPO_ROOT/.github/ci_commit_pins/nvidia_cutlass_dsl.txt"

if [[ ! -f "$PIN_FILE" ]]; then
  echo "error: pin file not found: $PIN_FILE" >&2
  exit 1
fi

CUTE_VERSION="$(grep -v '^[[:space:]]*\(#\|$\)' "$PIN_FILE" | head -n1 | tr -d '[:space:]')"

if [[ -z "$CUTE_VERSION" ]]; then
  echo "error: pin file '$PIN_FILE' is empty" >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  PIP_INSTALL=(uv pip install)
  PIP_UNINSTALL=(uv pip uninstall)
else
  PIP_INSTALL=(python -m pip install)
  PIP_UNINSTALL=(python -m pip uninstall -y)
fi

echo "==> Uninstalling any existing nvidia-cutlass-dsl* packages"
# Ignore errors so this works on a fresh environment.
"${PIP_UNINSTALL[@]}" \
  nvidia-cutlass-dsl \
  nvidia-cutlass-dsl-libs-base \
  nvidia-cutlass-dsl-libs-cu12 \
  nvidia-cutlass-dsl-libs-cu13 \
  || true

if [[ "$VARIANT" == "cu13" ]]; then
  echo "==> Installing nvidia-cutlass-dsl[cu13]==$CUTE_VERSION"
  "${PIP_INSTALL[@]}" --prerelease=allow \
    "nvidia-cutlass-dsl[cu13]==$CUTE_VERSION"

  # The PyPI nvidia-cutlass-dsl-libs-base and -libs-cu13 wheels ship several
  # Python files (tensor.py, algorithm.py, _nvvm_enum_gen.py, nvvm_wrappers.py,
  # ...) at the same install path with conflicting content. cu13's versions
  # match the cu13 binary; libs-base's versions either reference symbols
  # missing from cu13's arch or use older API signatures. When the resolver
  # lands them in non-deterministic order and libs-base wins on disk, every
  # cute test crashes. Force-reinstall cu13 *with --no-deps* so its files are
  # the last writers on disk.
  echo "==> Reinstalling nvidia-cutlass-dsl-libs-cu13 (no deps) so its files win"
  "${PIP_INSTALL[@]}" --prerelease=allow --reinstall --no-deps \
    "nvidia-cutlass-dsl-libs-cu13==$CUTE_VERSION"
else
  echo "==> Installing nvidia-cutlass-dsl==$CUTE_VERSION (cu12 / libs-base only)"
  "${PIP_INSTALL[@]}" --prerelease=allow \
    "nvidia-cutlass-dsl==$CUTE_VERSION"
fi

echo "==> Verifying install"
python -c "
import cutlass
version = getattr(cutlass, '__version__', None) or 'installed'
print(f'nvidia-cutlass-dsl: {version}')
"
