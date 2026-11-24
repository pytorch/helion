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
#   ./scripts/install_cute.sh           # verifies torch CUDA is 13.x
#   ./scripts/install_cute.sh cu13
#   ./scripts/install_cute.sh cu13 --install-only  # before rebuilding torch
#   ./scripts/install_cute.sh cu13 --verify-only   # after rebuilding torch
#
# Outside CI (no `uv` on PATH) this falls back to plain `python -m pip`.

set -euo pipefail

VARIANT=""
MODE=""
for arg in "$@"; do
  case "$arg" in
    --install-only|--verify-only)
      if [[ -n "$MODE" ]]; then
        echo "error: choose only one of --install-only and --verify-only" >&2
        exit 1
      fi
      MODE="$arg"
      ;;
    cu13)
      if [[ -n "$VARIANT" ]]; then
        echo "error: specify cu13 only once" >&2
        exit 1
      fi
      VARIANT="$arg"
      ;;
    *)
      echo "error: unsupported argument '$arg' (expected cu13, --install-only or --verify-only)" >&2
      exit 1
      ;;
  esac
done

# An editable torch may not import until its native library has been rebuilt.
# Never probe torch in install-only mode, even to choose the CUDA variant.
if [[ "$MODE" == --install-only && -z "$VARIANT" ]]; then
  echo "error: --install-only requires an explicit cu13 argument" >&2
  exit 1
fi

# Helion's cute backend requires CUDA 13.x. Verify the installed torch matches,
# either implicitly or via an explicit ``cu13`` argument.
if [[ -z "$VARIANT" ]]; then
  TORCH_CUDA="$(python -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null || echo "")"
  case "$TORCH_CUDA" in
    13.*|13)  VARIANT="cu13" ;;
    "")
      echo "error: torch is not installed or has no CUDA build; pass cu13 explicitly" >&2
      exit 1
      ;;
    *)
      echo "error: torch CUDA version '$TORCH_CUDA' is not supported (expected 13.x)" >&2
      exit 1
      ;;
  esac
  echo "==> Detected torch CUDA $TORCH_CUDA -> variant $VARIANT"
fi

verify_install() {
  echo "==> Verifying install"
  python -c "
import cutlass
version = getattr(cutlass, '__version__', None) or 'installed'
print(f'nvidia-cutlass-dsl: {version}')
import tvm_ffi
print(f'apache-tvm-ffi: {getattr(tvm_ffi, \"__version__\", \"installed\")}')
"
}

if [[ "$MODE" == --verify-only ]]; then
  verify_install
  exit 0
fi

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

echo "==> Installing nvidia-cutlass-dsl[cu13]==$CUTE_VERSION"
"${PIP_INSTALL[@]}" \
  "nvidia-cutlass-dsl[cu13]==$CUTE_VERSION" \
  "nvidia-cutlass-dsl-libs-base[sass]==$CUTE_VERSION"

echo "==> Installing apache-tvm-ffi (required for tcgen05 TVM FFI launch)"
"${PIP_INSTALL[@]}" apache-tvm-ffi

if [[ "$MODE" == --install-only ]]; then
  echo "==> Verification deferred; run scripts/install_cute.sh cu13 --verify-only after rebuilding torch"
else
  verify_install
fi
