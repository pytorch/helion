"""
Helion FP8 RowWise scaled_mm Example
====================================
FP8 (e4m3) RowWise scaled matrix multiplication on Helion:

    out[m, n] = scale_a[m] * scale_b[n] * sum_k a[m, k] * b[k, n]

``scale_a`` is passed as a ``(M, N)`` stride-(1,0) column-vector view (read as a
scalar per subtile in the fused tcgen05 epilogue) and ``scale_b`` as a rank-1
per-column row vector; this is the layout the CuTe backend's
fused-rowwise-scale epilogue recognizes.
"""

# %%
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import helion
from helion._testing import DEVICE
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable


# %%
@helion.kernel(static_shapes=True)
def fp8_matmul(
    x: torch.Tensor,
    y: torch.Tensor,
    scale_a2d: torch.Tensor,
    scale_b1d: torch.Tensor,
) -> torch.Tensor:
    """FP8 RowWise scaled_mm: ``out = scale_a[m] * scale_b[n] * (x @ y)`` in BF16.

    Args:
        x: Left input [m, k] in FP8 (e4m3), row-major.
        y: Right input [k, n] in FP8 (e4m3), column-major (K-contiguous) preferred.
        scale_a2d: Per-row scale as a ``(m, n)`` stride-(1,0) column-vector view.
        scale_b1d: Per-column scale as a rank-1 ``(n,)`` row vector.

    Returns:
        Output [m, n] in BF16.
    """
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f"size mismatch {k} != {k2}"
    out = torch.empty([m, n], dtype=torch.bfloat16, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
        # Fold the rowwise scale on the f32 accumulator before the bf16 cast.
        out[tile_m, tile_n] = (acc * scale_a2d[tile_m, tile_n] * scale_b1d[tile_n]).to(
            torch.bfloat16
        )
    return out


# %%
def reference_scaled_mm(
    x: torch.Tensor,
    y: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    """Reference using ``torch._scaled_mm`` (column-major second operand)."""
    if y.stride(0) == 1 and y.stride(1) > 1:
        y_col_major = y
    else:
        y_col_major = y.T.contiguous().T
    return torch._scaled_mm(
        x, y_col_major, scale_a, scale_b, use_fast_accum=False, out_dtype=torch.bfloat16
    )


# %%
def fp8_matmul_tritonbench(
    tb_op: object,
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> Callable[[], torch.Tensor]:
    """Wrapper for TritonBench: a [m, k] fp8, b [k, n] fp8 (col-major), rowwise
    scales scale_a [m, 1], scale_b [1, n]."""
    m, _ = a.size()
    _, n = b.size()
    scale_a2d = scale_a.reshape(m, 1).expand(m, n)
    scale_b1d = scale_b.reshape(n)
    return lambda: fp8_matmul(a, b, scale_a2d, scale_b1d)


# %%
def check(m: int, k: int, n: int) -> None:
    """Validate fp8_matmul against the torch._scaled_mm reference."""
    x = (torch.randn([m, k], device=DEVICE) * 0.4).to(torch.float8_e4m3fn)
    y = (torch.randn([k, n], device=DEVICE) * 0.4).to(torch.float8_e4m3fn)
    y = y.T.contiguous().T  # column-major (K-contiguous)
    scale_a = (torch.rand([m, 1], device=DEVICE) + 0.5).to(torch.float32)
    scale_b = (torch.rand([1, n], device=DEVICE) + 0.5).to(torch.float32)
    scale_a2d = scale_a.reshape(m, 1).expand(m, n)
    scale_b1d = scale_b.reshape(n)

    from helion._testing import run_example

    run_example(
        lambda a, b: fp8_matmul(a, b, scale_a2d, scale_b1d),
        lambda a, b: reference_scaled_mm(a, b, scale_a, scale_b),
        (x, y),
        atol=0.1,
        rtol=0.1,
    )


# %%
def main() -> None:
    check(64, 2048, 2048)


# %%
if __name__ == "__main__":
    main()
