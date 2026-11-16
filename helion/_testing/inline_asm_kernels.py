"""Small kernels exercising opaque effects across reduction sweeps."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(autotune_effort="none")
def overwrite_after_reduction(x: torch.Tensor, pointers: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for row in hl.tile(x.shape[0], block_size=1):
        value = x[row, :]
        total = value.sum(-1)
        hl.inline_asm_elementwise(
            "atom.global.exch.b32 $0, [$1], $2;",
            "=r,l,r",
            [pointers[row], total + 1],
            torch.int32,
            False,
            1,
        )
        out[row, :] = value - total[:, None]
    return out


@helion.kernel(autotune_effort="none")
def store_between_reductions(
    x: torch.Tensor, target: torch.Tensor, use_fresh: hl.constexpr
) -> torch.Tensor:
    if use_fresh:
        storage = torch.empty((x.shape[0] * 2,), dtype=x.dtype, device=x.device)
        destination = storage[::2]
    else:
        destination = target
    out = torch.empty((x.shape[0],), dtype=x.dtype, device=x.device)
    for row in hl.tile(x.shape[0], block_size=1):
        values = x[row, :]
        destination[row] = values.sum(-1)
        out[row] = destination[row] * 2 + values.amax(-1)
    return out


@helion.kernel(autotune_effort="none", static_shapes=True)
def overwritten_reduction_input(
    x: torch.Tensor, pointers: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(x)
    for row in hl.tile(x.shape[0], block_size=1):
        value = x[row, :]
        hl.inline_asm_elementwise(
            "atom.global.exch.b32 $0, [$1], $2;",
            "=r,l,r",
            [pointers[row, :], value + 1],
            torch.int32,
            False,
            1,
        )
        out[row, :] = value - value.sum(-1, keepdim=True)
    return out


@helion.kernel(autotune_effort="none", static_shapes=True)
def read_after_reduction_effect(
    x: torch.Tensor, pointers: torch.Tensor
) -> torch.Tensor:
    out = torch.empty((x.shape[0],), dtype=x.dtype, device=x.device)
    for row in hl.tile(x.shape[0], block_size=1):
        value = x[row, :]
        previous = hl.inline_asm_elementwise(
            "atom.global.exch.b32 $0, [$1], $2;",
            "=r,l,r",
            [pointers[row, :], value + 1],
            torch.int32,
            False,
            1,
        )
        total = previous.sum(-1)
        first = x[row, 0]
        out[row] = first + total
    return out
