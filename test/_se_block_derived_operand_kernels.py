"""The se_block backward GEMMs in their derived-operand form.

``examples/se_block.py`` used to form the shared matmul operand
``g = 2 * grad_out * x * s * (1 - s)`` inside each gradient's reduction
loop; the example now computes it once in a separate pointwise kernel.  The
CuTe tests that exercise a derived (transformed) operand with a seeded carry
(``se_block_bwd_dx``) and a transposed direct operand next to a derived one
(``se_block_bwd_dw``) keep those bodies here.
"""

from __future__ import annotations

import torch
from torch import Tensor

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def se_block_bwd_dx(grad_out: Tensor, x: Tensor, w: Tensor, s: Tensor) -> Tensor:
    """grad_x = 2 * grad_out * s + (2 * grad_out * x * s * (1 - s)) @ w.T with
    the elementwise factor formed in fp32, rounded once to ``x.dtype`` as the
    matmul's A operand, and fp32 accumulation seeded before the K loop."""
    m, n = x.size()

    grad_x = torch.empty([m, n], dtype=x.dtype, device=x.device)

    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        acc += 2.0 * grad_out[tile_m, tile_n] * s[tile_m, tile_n]

        for tile_k in hl.tile(n):
            grad_to_w = (
                2.0
                * grad_out[tile_m, tile_k].to(torch.float32)
                * x[tile_m, tile_k].to(torch.float32)
                * s[tile_m, tile_k].to(torch.float32)
                * (1.0 - s[tile_m, tile_k].to(torch.float32))
            ).to(x.dtype)
            acc = torch.addmm(acc, grad_to_w, w[tile_n, tile_k].T)

        grad_x[tile_m, tile_n] = acc.to(x.dtype)

    return grad_x


@helion.kernel(static_shapes=True)
def se_block_bwd_dw(grad_out: Tensor, x: Tensor, s: Tensor) -> Tensor:
    """grad_w = x.T @ (2 * grad_out * x * s * (1 - s)): the transposed input
    tile is the direct A operand, the derived fp16 factor the B operand."""
    m, n = x.size()

    grad_w = torch.empty([n, n], dtype=x.dtype, device=x.device)

    for tile_n1, tile_n2 in hl.tile([n, n]):
        acc_w = hl.zeros([tile_n1, tile_n2], dtype=torch.float32)
        for tile_m in hl.tile(m):
            grad_to_w = (
                2.0
                * grad_out[tile_m, tile_n2].to(torch.float32)
                * x[tile_m, tile_n2].to(torch.float32)
                * s[tile_m, tile_n2].to(torch.float32)
                * (1.0 - s[tile_m, tile_n2].to(torch.float32))
            ).to(x.dtype)
            acc_w = torch.addmm(acc_w, x[tile_m, tile_n1].T, grad_to_w)

        grad_w[tile_n1, tile_n2] = acc_w.to(x.dtype)

    return grad_w
