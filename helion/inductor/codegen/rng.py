"""Inductor's random ops that Helion has no language op for, written as the
torch ops Helion compiles, for HelionKernel to call from a kernel."""

from __future__ import annotations

import torch

from ...language.random_ops import _philox_uint32x4


def randint64(
    seed: torch.Tensor, offset: torch.Tensor, low: int, high: int
) -> torch.Tensor:
    """Inductor's ops.randint64, as its Triton kernels draw it
    (triton_helpers.randint64): the first two words r0 and r1 of Philox at
    the counter (offset as uint32, 0, 0, 0) with the key seed, read as the
    uint64 r0 | r1 << 32, taken modulo high - low and offset by low. A kernel
    calls it on the seed and the element's flat index,
        tmp0 = in0[()]                          # seeds[i] (ops.load_seed)
        tmp1 = (tile_x.index).to(torch.int32)   # offset
        tmp2 = randint64(tmp0, tmp1, 0, 44100)
    and Helion inlines the ten Philox rounds into the kernel. The uint64
    remainder is computed in int64 as (r1 * 2**32 + r0) mod size, from the
    remainders of r1 and r0, whose products stay below 2**63 for a size of up
    to 2**31 (HelionKernel.randint64() declines larger ones)."""
    c0, c1, _, _ = _philox_uint32x4(seed, offset.to(torch.int64) & 0xFFFFFFFF)
    size = high - low
    return (c1 % size * (2**32 % size) + c0 % size) % size + low
