"""The Metal benchmark's Helion kernel for int4_gemm at a few rows (see
int4_gemm.py).

The example's kernel unpacks each int8 into two bf16 weights (mask, compare,
select, two conversions) and contracts them with the activations.  At a few
rows that decode, not memory, bounds it: 1 x 3584 x 8192 ran at 114 GB/s on
an M4 Pro, where MLX's ``quantized_matmul`` reads its weights at 340 GB/s.

This kernel reads the same packed weights as int32 words (four columns of
one packed row each, a zero-copy view) and decodes them the way MLX's
``qdot`` does:

- one XOR with ``0x88888888`` turns every signed nibble ``v`` into ``v + 8``;
- each nibble is masked in place, not shifted down: nibble ``j`` of a word
  is the integer ``(v + 8) * 2**(8 j)`` (the high nibbles after one shift by
  4), which converts to float exactly;
- the activations are pre-scaled by ``2**(-8 j)`` instead, exactly, once per
  packed row for every column;
- the ``+ 8`` comes off at the end as ``8 * sum(x)`` per row.

That is a mask, a conversion and a multiply-add per weight.  The products
contract with ``hl.dot``, whose K tile spreads over the threads, the word
columns on the threads beside them, so one row's words load as contiguous
runs.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl

#: Most rows int4_gemm runs on this kernel; more run the example's, whose
#: matmul lowers on MPP.  A Metal matmul block takes at least 16 rows
#: (``min_dot_size``), so at 4 rows this kernel's scalar dots compute 16 and
#: ran 2.2x slower than the example (800 µs against 370 µs at 4 x 3584 x
#: 8192).
GEMV_MAX_ROWS = 1


@helion.kernel(static_shapes=True)
def int4_gemv(a: torch.Tensor, w4: torch.Tensor) -> torch.Tensor:
    """``a`` [M, K] bf16 times int4 weights in [-8, 7] packed two per byte
    along K, row 2i in each low nibble, read as int32 words [K/2, N/4]."""
    m, k = a.size()
    kp, n4 = w4.size()
    f32 = torch.float32
    out = torch.empty([m, n4, 4], dtype=a.dtype, device=a.device)
    for tile_m, tile_c in hl.tile([m, n4]):
        acc0 = hl.zeros([tile_m, tile_c], dtype=f32)
        acc1 = hl.zeros([tile_m, tile_c], dtype=f32)
        acc2 = hl.zeros([tile_m, tile_c], dtype=f32)
        acc3 = hl.zeros([tile_m, tile_c], dtype=f32)
        # sum(x) over K, in every column: the bias of the + 8.  A dot, not a
        # .sum(): a reduction takes thread axis 0 (reduction_axis_first), off
        # the word columns, whose loads then don't coalesce.
        xs = hl.zeros([tile_m, tile_c], dtype=f32)
        for tile_p in hl.tile(kp):
            u = w4[tile_p, tile_c] ^ -0x77777778  # 0x88888888
            uh = u >> 4
            xl = a[tile_m, tile_p.index * 2].to(f32)
            xh = a[tile_m, tile_p.index * 2 + 1].to(f32)
            xs = hl.dot(xl + xh, hl.full([tile_p, tile_c], 1.0, dtype=f32), acc=xs)
            acc0 = hl.dot(xl, (u & 0xF).to(f32), acc=acc0)
            acc0 = hl.dot(xh, (uh & 0xF).to(f32), acc=acc0)
            acc1 = hl.dot(xl * 2.0**-8, (u & 0xF00).to(f32), acc=acc1)
            acc1 = hl.dot(xh * 2.0**-8, (uh & 0xF00).to(f32), acc=acc1)
            acc2 = hl.dot(xl * 2.0**-16, (u & 0xF0000).to(f32), acc=acc2)
            acc2 = hl.dot(xh * 2.0**-16, (uh & 0xF0000).to(f32), acc=acc2)
            acc3 = hl.dot(xl * 2.0**-24, (u & 0xF000000).to(f32), acc=acc3)
            acc3 = hl.dot(xh * 2.0**-24, (uh & 0xF000000).to(f32), acc=acc3)
        out[tile_m, tile_c, 0] = (acc0 - 8.0 * xs).to(a.dtype)
        out[tile_m, tile_c, 1] = (acc1 - 8.0 * xs).to(a.dtype)
        out[tile_m, tile_c, 2] = (acc2 - 8.0 * xs).to(a.dtype)
        out[tile_m, tile_c, 3] = (acc3 - 8.0 * xs).to(a.dtype)
    return out.view(m, n4 * 4)
