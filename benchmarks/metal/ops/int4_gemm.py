"""int4_gemm: bf16 activations times int4 weights in [-8, 6], no scales.

The weights are packed outside timing:

- Helion and torch get what the nightly gives Helion
  (``int4_gemm_tritonbench``): two int4 values per int8 along K, row 2i in
  the low nibble, row-major [K/2, N].  Torch eager is the example's
  ``reference_matmul_bf16_int4``, which unpacks them for a bf16 matmul.
- MLX gets its native 4-bit format (``w + 8`` with scale 1 and bias -8), the
  layout ``mx.quantized_matmul`` reads.

The reference runs in float32 on the GPU: the largest rows are 5-15 TFLOP,
and float32 accumulation of these integer-weighted sums is exact far below
bf16 ulp.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from helion._testing import import_example

_examples_int4_gemm = import_example("int4_gemm")
matmul_bf16_int4 = _examples_int4_gemm.matmul_bf16_int4
reference_matmul_bf16_int4 = _examples_int4_gemm.reference_matmul_bf16_int4

if TYPE_CHECKING:
    import mlx.core as mx

    from ..manifest import Case

MODE = "fwd"
HELION_KERNELS = (matmul_bf16_int4,)
GROUP = 64  # quantization group of the MLX format (scale 1)


def inputs(case: Case, gen: torch.Generator) -> dict[str, torch.Tensor]:
    b, seq, k, n = (case.params[key] for key in ("B", "L", "K", "N"))
    w = torch.randint(-8, 7, (k, n), generator=gen, dtype=torch.int32)
    pairs = w.to(torch.int8).reshape(k // 2, 2, n)
    packed = (pairs[:, 0] & 0xF) | (pairs[:, 1] << 4)
    q = (w.T + 8).to(torch.uint8)  # [N, K] in [0, 15]
    nibbles = q.to(torch.int64).reshape(n, k // 8, 8)
    words = (nibbles << (4 * torch.arange(8))).sum(-1)
    return {
        "x": torch.randn(b, seq, k, generator=gen, dtype=torch.bfloat16),
        "w_packed": packed,
        "w_mlx": torch.from_numpy(words.numpy().astype(np.uint32)),
        "scales": torch.ones(n, k // GROUP, dtype=torch.bfloat16),
        "biases": torch.full((n, k // GROUP), -8.0, dtype=torch.bfloat16),
    }


def _unpack(w_packed: torch.Tensor) -> torch.Tensor:
    """[K/2, N] int8 -> [K, N] bf16.  Masks before shifting, so no int8 shift
    overflows (Inductor on MPS miscompiles ``(w << 4) >> 4``)."""
    lo = ((w_packed & 0xF) ^ 8) - 8
    hi = (((w_packed >> 4) & 0xF) ^ 8) - 8
    k2, n = w_packed.shape
    return torch.stack([lo, hi], dim=1).reshape(2 * k2, n).to(torch.bfloat16)


def reference(
    x: torch.Tensor, w_packed: torch.Tensor, *mlx_only: torch.Tensor
) -> torch.Tensor:
    x2d = x.reshape(-1, x.shape[-1]).to("mps", torch.float32)
    return (x2d @ _unpack(w_packed).to("mps", torch.float32)).cpu()


def eager(
    x: torch.Tensor, w_packed: torch.Tensor, *mlx_only: torch.Tensor
) -> torch.Tensor:
    return reference_matmul_bf16_int4(x.reshape(-1, x.shape[-1]), w_packed)


def helion(
    x: torch.Tensor, w_packed: torch.Tensor, *mlx_only: torch.Tensor
) -> torch.Tensor:
    return matmul_bf16_int4(x.reshape(-1, x.shape[-1]), w_packed)


def mlx(
    x: mx.array, w_packed: mx.array, w_mlx: mx.array, scales: mx.array, biases: mx.array
) -> mx.array:
    import mlx.core as mx

    return mx.quantized_matmul(
        x.reshape(-1, x.shape[-1]),
        w_mlx,
        scales,
        biases,
        transpose=True,
        group_size=GROUP,
        bits=4,
    )
