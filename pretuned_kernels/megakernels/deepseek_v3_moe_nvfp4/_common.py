"""Shared device-side building blocks for the DeepSeek NVFP4 kernels."""

from __future__ import annotations

import operator

import torch

import helion.language as hl

FP4_MAX = 6.0
MMA_N = 16


def fp4_nibble(value: torch.Tensor) -> torch.Tensor:
    """Round scaled values to E2M1 with FlashInfer's tie convention."""
    magnitude = torch.abs(value)
    code = (magnitude > 0.25).to(torch.int32)
    # Keep these additions functional: aten.add_ returns an aliased InputBuffer,
    # which the shared Inductor-to-Helion lowering cannot consume as pointwise IR.
    code = code + (magnitude >= 0.75).to(torch.int32)
    code = code + (magnitude > 1.25).to(torch.int32)
    code = code + (magnitude >= 1.75).to(torch.int32)
    code = code + (magnitude > 2.5).to(torch.int32)
    code = code + (magnitude >= 3.5).to(torch.int32)
    code = code + (magnitude > 5.0).to(torch.int32)
    return code | ((value < 0).to(torch.int32) << 3)


def first_mma_column(value: torch.Tensor) -> torch.Tensor:
    """Select the real token column from the padded native-MMA N tile."""
    column = hl.arange(MMA_N)
    return torch.sum(value * (column[None, :] == 0).to(torch.float32), dim=-1)


def stable_argmax_id(
    values: torch.Tensor,
    indices: torch.Tensor,
    sentinel: int,
) -> torch.Tensor:
    """Select the lowest index among equal maxima."""
    maximum = torch.amax(values, dim=-1, keepdim=True)
    return torch.amin(
        torch.where(
            values == maximum,
            indices,
            torch.full_like(indices, sentinel),
        ),
        dim=-1,
    )


def take_stable_argmax(
    values: torch.Tensor,
    indices: torch.Tensor,
    sentinel: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select one stable maximum and mask it from the next selection."""
    selected = stable_argmax_id(values, indices, sentinel)
    remaining = torch.where(
        indices == selected[:, None],
        torch.full_like(values, float("-inf")),
        values,
    )
    return selected, remaining


def take_stable_top8(
    values: torch.Tensor,
    indices: torch.Tensor,
    sentinel: int,
) -> tuple[torch.Tensor, ...]:
    """Select eight stable maxima while keeping every value register-local."""
    id_0, values = take_stable_argmax(values, indices, sentinel)
    id_1, values = take_stable_argmax(values, indices, sentinel)
    id_2, values = take_stable_argmax(values, indices, sentinel)
    id_3, values = take_stable_argmax(values, indices, sentinel)
    id_4, values = take_stable_argmax(values, indices, sentinel)
    id_5, values = take_stable_argmax(values, indices, sentinel)
    id_6, values = take_stable_argmax(values, indices, sentinel)
    id_7, _values = take_stable_argmax(values, indices, sentinel)
    return id_0, id_1, id_2, id_3, id_4, id_5, id_6, id_7


def swizzle_scales(scale: torch.Tensor) -> torch.Tensor:
    """[..., rows, groups] E4M3 block scales -> [..., rows/128 * groups/32, 16, 256]:
    per 128-row tile and K step of 32 groups (row-tile major), the 4 KB in
    tcgen05's 32x4x4 scale-atom order, as cuBLASLt and TRT-LLM store NVFP4 scales."""
    *batch, rows, groups = scale.shape
    n = len(batch)
    tiles = scale.reshape(*batch, rows // 128, 4, 32, groups // 32, 8, 4)
    # (tile, j, i, step, c, k) -> (tile, step, c, i, j, k): row j * 32 + i.
    order = [*range(n), n, n + 3, n + 4, n + 2, n + 1, n + 5]
    return (
        tiles.permute(order)
        .reshape(*batch, rows // 128 * (groups // 32), 16, 256)
        .contiguous()
    )


def scale_tile(chunk: torch.Tensor) -> torch.Tensor:
    """[16, 256] swizzled scale bytes of a 128-row tile and K step -> [128, 32]."""
    return chunk.reshape(8, 32, 4, 4).permute(2, 1, 0, 3).reshape(128, 32)


def _popcount(v: torch.Tensor) -> torch.Tensor:
    """Set bits per int32 lane."""
    v = v - ((v >> 1) & 0x55555555)
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
    return (((v + (v >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24


def distinct_table(
    ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The routed experts (of 256) as bitmap words [8], each word's exclusive
    prefix count, and the distinct count."""
    lanes = hl.arange(8)
    row = ids.unsqueeze(0)
    bits = torch.where((row >> 5) == lanes.unsqueeze(1), 1 << (row & 31), 0)
    words = hl.reduce(operator.or_, bits, dim=1)
    counts = _popcount(words)
    before = hl.cumsum(counts, dim=0) - counts
    return words, before, hl.reduce(operator.add, counts, dim=0)


def expert_of(words: torch.Tensor, before: torch.Tensor, j: object) -> torch.Tensor:
    """The j-th distinct expert (ascending)."""
    counts = _popcount(words)
    hit = (before <= j) & (before + counts > j)
    word = hl.reduce(operator.or_, torch.where(hit, words, 0), dim=0)
    key = hl.reduce(operator.or_, torch.where(hit, before * 8 + hl.arange(8), 0), dim=0)
    # int32 lanes: an int64 index (large weights) would widen the popcount.
    upto = _popcount(word & ((2 << hl.arange(32).to(torch.int32)) - 1))
    bit = hl.reduce(operator.add, (upto <= j - (key >> 3)).to(torch.int32), dim=0)
    return (key & 7) * 32 + bit
