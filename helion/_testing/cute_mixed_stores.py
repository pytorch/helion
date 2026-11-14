"""Matmul epilogues whose output stores differ in element type or skip the accumulator.

``bmm_three_stores`` is the store set Inductor emits for ``bmm`` followed by
``out1 = aux.float()`` and ``out2 = (acc + out1).bfloat16()``: an FP32 store
of the accumulator, an FP32 copy of an auxiliary input that never reads the
accumulator (an "aux-only" store) and a BF16 store of their sum into a
permuted view. ``mm_mixed_stores`` is its rank-2, aux-only-free core: an FP32
accumulator store beside a BF16 ``acc + aux`` store. ``mm_and_copy`` puts an
aux-only ``2 * aux`` store first or last, where it waits for or releases the
accumulator. ``mm_reread_output`` copies an output the epilogue writes
itself, which the epilogue must not adopt.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def bmm_three_stores(
    a: torch.Tensor,
    b: torch.Tensor,
    aux: torch.Tensor,
    acc_out: torch.Tensor,
    aux_out: torch.Tensor,
    sum_out: torch.Tensor,
) -> None:
    batch, m, k = a.shape
    n = b.shape[2]
    for tile_b, tile_m, tile_n in hl.tile([batch, m, n]):
        acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.baddbmm(
                acc, a[tile_b, tile_m, tile_k], b[tile_b, tile_k, tile_n]
            )
        aux_f32 = aux[tile_b, tile_m, tile_n].to(torch.float32)
        acc_out[tile_b, tile_m, tile_n] = acc
        aux_out[tile_b, tile_m, tile_n] = aux_f32
        sum_out[tile_b, tile_m, tile_n] = (acc + aux_f32).to(torch.bfloat16)


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


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def mm_and_copy(
    a: torch.Tensor,
    b: torch.Tensor,
    aux: torch.Tensor,
    acc_out: torch.Tensor,
    aux_out: torch.Tensor,
    copy_first: hl.constexpr,
) -> None:
    m, k = a.shape
    n = b.shape[1]
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        if copy_first:
            aux_out[tile_m, tile_n] = aux[tile_m, tile_n].to(torch.float32) * 2.0
            acc_out[tile_m, tile_n] = acc
        else:
            acc_out[tile_m, tile_n] = acc
            aux_out[tile_m, tile_n] = aux[tile_m, tile_n].to(torch.float32) * 2.0


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def mm_reread_output(
    a: torch.Tensor, b: torch.Tensor, acc_out: torch.Tensor, copy_out: torch.Tensor
) -> None:
    m, k = a.shape
    n = b.shape[1]
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        acc_out[tile_m, tile_n] = acc.to(torch.bfloat16)
        copy_out[tile_m, tile_n] = acc_out[tile_m, tile_n]


def bmm_three_stores_args(
    batch: int, m: int, n: int, k: int, *, device: str, random: bool
) -> tuple[torch.Tensor, ...]:
    """``bmm_three_stores`` operands; ``sum_out`` is the ``(batch, m, n)``
    view of a row-major ``(m, batch, n)`` buffer, as Inductor permutes it."""
    bf16 = torch.bfloat16

    def tensor(*shape: int, dtype: torch.dtype = bf16) -> torch.Tensor:
        if random:
            return torch.randn(shape, device=device).to(dtype)
        return torch.empty(shape, device=device, dtype=dtype)

    return (
        tensor(batch, m, k),
        tensor(batch, k, n),
        tensor(batch, m, n),
        torch.empty((batch, m, n), device=device, dtype=torch.float32),
        torch.empty((batch, m, n), device=device, dtype=torch.float32),
        torch.empty((m, batch, n), device=device, dtype=bf16).permute(1, 0, 2),
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


def mm_and_copy_args(
    m: int, n: int, k: int, *, device: str, random: bool
) -> tuple[torch.Tensor, ...]:
    """``mm_and_copy`` operands, without its ``copy_first`` flag."""
    a, b, aux, acc_out, _sum_out = mm_mixed_stores_args(
        m, n, k, device=device, random=random
    )
    return a, b, aux, acc_out, torch.empty_like(acc_out)
