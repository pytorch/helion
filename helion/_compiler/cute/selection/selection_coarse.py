"""Guarded coarse-rank selection as composable Helion tensor expressions.

The first axis contains consecutive sets of candidate groups. The caller can
place all groups sharing a collective execution domain in this axis: the
fallback predicate reduces over the complete axis, so the conditional is
uniform over that domain. No thread identifiers or shuffle operations occur
in this algorithm.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch
from torch._inductor.decomposition import select_decomp_table
from torch.fx.experimental.proxy_tensor import make_fx

from .selection_network import selection_network

if TYPE_CHECKING:
    from torch.fx import GraphModule


def _ranks(keys: torch.Tensor, index_bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    if keys.dtype == torch.int32:
        return keys, keys != -2147483648
    return (keys >> index_bits).to(torch.int32), keys != -9223372036854775808


def _positions(
    keys: torch.Tensor, vector_width: int, groups_per_result: int
) -> torch.Tensor:
    group = torch.arange(keys.shape[0], device=keys.device)[:, None]
    register = torch.arange(keys.shape[1], device=keys.device)
    return (
        ((register // vector_width) * groups_per_result + group % groups_per_result)
        * vector_width
        + register % vector_width
    ).to(torch.int32)


def coarse_rank_keys(
    keys: torch.Tensor,
    index_bits: int,
    vector_width: int,
    groups_per_result: int,
) -> torch.Tensor:
    """Construct finite positive coarse ranks with reversed column payloads."""
    rank, live = _ranks(keys, index_bits)
    mask = (1 << index_bits) - 1
    if keys.dtype == torch.int32:
        payload = mask - _positions(keys, vector_width, groups_per_result)
    else:
        payload = (keys & mask).to(torch.int32)
    valid = live & (rank >= 0x00800000) & (rank < 0x7F800000)
    coarse = ((rank & (-mask - 1)) | payload).view(torch.float32)
    return torch.where(valid, coarse, -float("inf"))


def coarse_rank_full_keys(
    keys: torch.Tensor,
    index_bits: int,
    vector_width: int,
    groups_per_result: int,
) -> torch.Tensor:
    """Retain the original exact ranks and deterministic index tie breaking."""
    if keys.dtype == torch.int64:
        return keys.clone()
    rank, live = _ranks(keys, index_bits)
    mask = (1 << index_bits) - 1
    column = _positions(keys, vector_width, groups_per_result)
    full = (rank.to(torch.int64) << index_bits) | (mask - column).to(torch.int64)
    return torch.where(live, full, -9223372036854775808)


def coarse_rank_guard(
    keys: torch.Tensor,
    selected: torch.Tensor,
    k: int,
    index_bits: int,
    groups_per_result: int,
    payload_only: bool = False,
) -> torch.Tensor:
    """Return one fallback flag for the entire collective execution domain."""
    groups, registers = keys.shape
    selected_registers = selected.shape[1]
    mask = (1 << index_bits) - 1
    group = torch.arange(groups, device=keys.device)[:, None]
    base = (group // groups_per_result) * groups_per_result
    cutoff_group = base + (k - 1) % groups_per_result
    cutoff_index = cutoff_group * selected_registers + (k - 1) // groups_per_result
    cutoff = torch.gather(selected.reshape(-1), 0, cutoff_index.reshape(-1))
    cutoff = cutoff.reshape(groups, 1).view(torch.int32) & (-mask - 1)

    rank, live = _ranks(keys, index_bits)
    matches = (live & ((rank & (-mask - 1)) == cutoff)).to(torch.int32)
    matches = matches.reshape(-1, groups_per_result * registers).sum(
        dim=1, dtype=torch.int32
    )
    unsupported = (live & (rank >= 0x7F800000)).reshape(-1).any()
    cutoff_valid = (cutoff >= 0x00800000) & (cutoff < 0x7F800000)
    fallback = (matches != 1).any() | unsupported | (~cutoff_valid).reshape(-1).any()

    if payload_only:
        slot = torch.arange(selected_registers, device=keys.device)
        position = slot * groups_per_result + group % groups_per_result
        previous_group = base + ((group - 1) & (groups_per_result - 1))
        previous_slot = torch.clamp(
            slot - (group % groups_per_result == 0).to(torch.int64), min=0
        )
        previous_index = previous_group * selected_registers + previous_slot
        buckets = selected.view(torch.int32) & (-mask - 1)
        previous = torch.gather(buckets.reshape(-1), 0, previous_index.reshape(-1))
        previous = previous.reshape(groups, selected_registers)
        collisions = (position > 0) & (position < k) & (buckets == previous)
        fallback = fallback | collisions.reshape(-1).any()
    return fallback


def coarse_rank_recover(
    keys: torch.Tensor,
    selected: torch.Tensor,
    k: int,
    index_bits: int,
    vector_width: int,
    groups_per_result: int,
    recovery: str = "direct",
) -> torch.Tensor:
    """Gather exact resident ranks for a proven selected set without reloading."""
    groups, registers = keys.shape
    mask = (1 << index_bits) - 1
    group = torch.arange(groups, device=keys.device)[:, None]
    bits = selected.view(torch.int32)
    column = mask - (bits & mask)
    owner = (column // vector_width) % groups_per_result
    owner = owner + (group // groups_per_result) * groups_per_result
    register = (column // (groups_per_result * vector_width)) * vector_width
    register = register + column % vector_width
    rank, _live = _ranks(keys, index_bits)
    if recovery == "packed":
        per_word = min(registers, 1 << ((32 // index_bits).bit_length() - 1))
        packed_registers = registers // per_word
        residues = (rank & mask).reshape(groups, packed_registers, per_word)
        shift = torch.arange(per_word, device=keys.device) * index_bits
        packed = (residues << shift.to(torch.int32)).sum(dim=2, dtype=torch.int32)
        index = owner * packed_registers + register // per_word
        word = torch.gather(packed.reshape(-1), 0, index.reshape(-1)).reshape(
            selected.shape
        )
        residue = (word >> ((register % per_word) * index_bits).to(torch.int32)) & mask
        exact_rank = (bits & (-mask - 1)) | residue
    else:
        index = owner * registers + register
        exact_rank = torch.gather(rank.reshape(-1), 0, index.reshape(-1)).reshape(
            selected.shape
        )
    full = (exact_rank.to(torch.int64) << index_bits) | (mask - column).to(torch.int64)
    slot = torch.arange(selected.shape[1], device=keys.device)
    unique = slot * groups_per_result + group % groups_per_result < k
    return torch.where(unique, full, -9223372036854775808)


def coarse_rank_selection(
    keys: torch.Tensor,
    k: int,
    sort_network: str,
    merge_schedule: str,
    index_bits: int,
    vector_width: int,
    groups_per_result: int,
    recovery: str = "direct",
    payload_only: bool = False,
) -> torch.Tensor:
    """Select exact keys, speculating on narrower ranks behind an exact guard.

    Choosing a complete warp's candidate groups as the first dimension makes
    the fallback branch uniform even when several rows share that warp. The
    individual stage functions are also callable in Helion kernels using a
    native Helion ``if`` around the scalar result of ``coarse_rank_guard``.
    """
    assert keys.dtype in (torch.int32, torch.int64)
    assert 0 < index_bits <= 23
    assert recovery in ("direct", "packed")
    assert vector_width > 0 and vector_width & (vector_width - 1) == 0
    assert keys.shape[1] % vector_width == 0
    coarse = coarse_rank_keys(keys, index_bits, vector_width, groups_per_result)
    selected = selection_network(
        coarse, k, sort_network, merge_schedule, "distributed", groups_per_result
    )
    fallback = coarse_rank_guard(
        keys, selected, k, index_bits, groups_per_result, payload_only
    )

    def exact(keys: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
        full = coarse_rank_full_keys(keys, index_bits, vector_width, groups_per_result)
        return selection_network(
            full, k, sort_network, merge_schedule, "distributed", groups_per_result
        )

    def refine(keys: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
        if payload_only:
            return (selected.view(torch.int32) & ((1 << index_bits) - 1)).to(
                torch.int64
            )
        refined = coarse_rank_recover(
            keys, selected, k, index_bits, vector_width, groups_per_result, recovery
        )
        return selection_network(
            refined, k, sort_network, merge_schedule, "distributed", groups_per_result
        )

    return torch.cond(fallback, exact, refine, (keys, selected))


@functools.cache
def trace_coarse_rank_selection(
    groups: int,
    registers: int,
    k: int,
    dtype: torch.dtype,
    sort_network: str,
    merge_schedule: str,
    index_bits: int,
    vector_width: int,
    groups_per_result: int,
    recovery: str = "direct",
    payload_only: bool = False,
) -> GraphModule:
    """Trace the conditional tensor algorithm without native kernel compilation."""

    def program(keys: torch.Tensor) -> torch.Tensor:
        return coarse_rank_selection(
            keys,
            k,
            sort_network,
            merge_schedule,
            index_bits,
            vector_width,
            groups_per_result,
            recovery,
            payload_only,
        )

    decompositions = select_decomp_table().copy()
    decompositions.pop(torch.ops.aten.fmin.default, None)
    decompositions.pop(torch.ops.aten.fmax.default, None)
    return make_fx(program, decomposition_table=decompositions)(
        torch.ones((groups, registers), dtype=dtype, device="cpu")
    )
