"""Autotune a helion matmul kernel on Modal H100s (no local GPU needed).

Uses helion's real autotuner with ModalSearch to find the best config,
then prints the copy-pasteable @helion.kernel decorator.

Usage:
    python benchmarks/bench_modal_matmul.py
"""

from __future__ import annotations

import os
import sys

# Ensure helion is importable
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Force ModalSearch as the autotuner
os.environ["HELION_AUTOTUNER"] = "ModalSearch"

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k2, n = y.size()
    assert k == k2
    out = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


def main() -> None:
    M, K, N = 4096, 4096, 4096
    print(f"Autotuning helion matmul: {M}x{K} @ {K}x{N} (float16)")
    print(f"Using ModalSearch -> dispatching to Modal H100 workers")
    print()

    # Create tensors on CPU — ModalSearch serializes them and
    # workers deserialize onto CUDA
    x = torch.randn(M, K, dtype=torch.float16)
    y = torch.randn(K, N, dtype=torch.float16)

    import logging

    # Filter out the compile_config traceback that fires when set_config()
    # tries to import triton locally (not available on Mac). The autotuner
    # prints the decorator at level INFO+5 — we want that but not the
    # WARNING-level compile error traceback.
    class _SuppressCompileError(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "triton codegen error" not in record.getMessage()

    logging.getLogger("helion").addFilter(_SuppressCompileError())

    try:
        best_config = matmul.autotune((x, y), force=True)
        print(f"\nBest config object: {best_config}")
    except (ModuleNotFoundError, Exception):
        # On machines without triton (e.g. Mac), autotune() finds the best
        # config and prints the decorator, but set_config() fails trying to
        # compile locally. The decorator was already printed above.
        pass


if __name__ == "__main__":
    main()
