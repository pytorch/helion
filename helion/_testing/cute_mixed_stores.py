"""Matmul epilogues whose output stores differ in element type.

``mm_mixed_stores`` is the rank-2 core of the store set Inductor emits for
``bmm`` followed by ``out2 = (acc + aux).bfloat16()`` beside the FP32
accumulator: an FP32 accumulator store and a BF16 ``acc + aux`` store.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def mm_mixed_stores(
    a: torch.Tensor,
    b: torch.Tensor,
    aux: torch.Tensor,
    acc_out: torch.Tensor,
    sum_out: torch.Tensor,
) -> None:
    m, k = a.shape
    n = b.shape[1]
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        acc_out[tile_m, tile_n] = acc
        sum_out[tile_m, tile_n] = (acc + aux[tile_m, tile_n].to(torch.float32)).to(
            torch.bfloat16
        )


def mm_mixed_stores_args(
    m: int, n: int, k: int, *, device: str, random: bool
) -> tuple[torch.Tensor, ...]:
    bf16 = torch.bfloat16

    def tensor(*shape: int) -> torch.Tensor:
        if random:
            return torch.randn(shape, device=device).to(bf16)
        return torch.empty(shape, device=device, dtype=bf16)

    return (
        tensor(m, k),
        tensor(k, n),
        tensor(m, n),
        torch.empty((m, n), device=device, dtype=torch.float32),
        torch.empty((m, n), device=device, dtype=bf16),
    )
