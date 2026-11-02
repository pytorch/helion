"""Resource and signed-index contracts for paired prefix-adaptive grouped GEMM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cute_mma import Tcgen05GroupedWorklistAnalysis


@dataclass(frozen=True)
class GroupedPrefixNBranch:
    n: int
    ab_stages: int
    c_stages: int
    a_bytes: int
    b_bytes: int
    d_bytes: int
    a_offset: int
    b_offset: int
    ab_barriers_offset: int
    d_offset: int
    arena_bytes: int
    # The M256 recipe validates the ordinary native allocation above, then
    # places D/A/B in one exclusive operand arena after a shared header.
    packed_a_offset: int | None = None
    packed_b_offset: int | None = None
    packed_d_offset: int | None = None


@dataclass(frozen=True)
class GroupedPrefixAdaptiveNProfile:
    primary: GroupedPrefixNBranch
    alternate: GroupedPrefixNBranch
    arena_bytes: int
    compiler_smem_bytes: int
    tmem_columns: int = 256
    common_d_tile: tuple[int, int] = (64, 64)
    row_tile: int = 128
    block_k: int = 64
    layout_kind: str = "whole_arena"


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def grouped_prefix_adaptive_n_input_domain(
    groups: int, m: int, n: int, k: int, *, source_m_tile: int = 128
) -> bool:
    """Bound every prefix, descriptor feature tile and terminal work index.

    The packed-A allocation supplies the strongest M bound. Even if all groups
    independently span it, G<=32 and F<=8 keep total work below 2**27; a
    launched hardware cluster stride cannot overflow the signed index domain.
    """
    maximum = (1 << 31) - 1
    if source_m_tile == 256:
        feature_tiles = max((n + 223) // 224, (n + 255) // 256)
        return (
            0 < groups <= 32
            and min(m, n, k) > 0
            and k >= 64
            and k % 64 == 0
            and m * k <= maximum
            and groups * n * k <= maximum
            and m * n <= maximum
            and m + 255 <= maximum
            and n % 32 == 0
            and feature_tiles <= 8
            and groups * ((m + 255) // 256) * feature_tiles <= 67_108_864
        )
    if source_m_tile != 128:
        return False
    return (
        0 < groups <= 32
        and min(m, n, k) > 0
        and k >= 64
        and k % 64 == 0
        and m * k <= maximum
        and groups * n * k <= maximum
        and m * n <= maximum
        and m + 127 <= maximum
        and n % 192 == n % 256 == 0
        and max(n // 192, n // 256) <= 8
        and groups * ((m + 127) // 128) * max(n // 192, n // 256) <= 67_108_864
    )


def resolve_grouped_prefix_adaptive_n_profile(
    group_count: int,
    dtype_bytes: int,
    output_dtype_bytes: int,
    capacity_bytes: int,
    *,
    source_m_tile: int = 128,
) -> GroupedPrefixAdaptiveNProfile | None:
    """Pick the deepest complete rings that fit both exclusive typed cores.

    The immutable descriptor recipe shares one byte arena. Preserve the
    measured alignment gaps and reserve the compiler's initial 1024-byte SMEM
    padding when checking the physical allocation. No branch allocations add.
    """
    if not (
        0 < group_count <= 32
        and dtype_bytes == 2
        and output_dtype_bytes == 2
        and capacity_bytes > 0
    ):
        return None

    common_end = _align(9 * 4, 16) + group_count * 4 * 4
    common_end = _align(common_end, 16) + group_count * 4
    holding = _align(common_end, 4)
    dealloc = _align(holding + 4, 8)
    acc = _align(dealloc + 8, 8)
    # Keep the now-unused scheduler-barrier gap so byte offsets agree with
    # the ordinary validated resource plan used to form each typed core.
    scheduler_gap = _align(acc + 4 * 8, 8)
    a_offset = _align(scheduler_gap + 2 * 8, 128)
    if source_m_tile == 256:
        return _resolve_m256_profile(
            group_count, dtype_bytes, output_dtype_bytes, capacity_bytes, a_offset
        )
    if source_m_tile != 128:
        return None
    branches: list[GroupedPrefixNBranch] = []
    for physical_n, c_stages in ((192, 3), (256, 4)):
        for ab_stages in range(32, 1, -1):
            a_bytes = 128 * 64 * dtype_bytes * ab_stages // 2
            b_bytes = physical_n * 64 * dtype_bytes * ab_stages // 2
            d_bytes = 64 * 64 * output_dtype_bytes * c_stages
            b_offset = _align(a_offset + a_bytes, 128)
            ab_barriers = _align(b_offset + b_bytes, 8)
            d_offset = _align(ab_barriers + 2 * ab_stages * 8, 1024)
            end = d_offset + d_bytes
            if end + 1024 <= capacity_bytes:
                branches.append(
                    GroupedPrefixNBranch(
                        physical_n,
                        ab_stages,
                        c_stages,
                        a_bytes,
                        b_bytes,
                        d_bytes,
                        a_offset,
                        b_offset,
                        ab_barriers,
                        d_offset,
                        end,
                    )
                )
                break
        else:
            return None
    arena_bytes = max(branch.arena_bytes for branch in branches)
    return GroupedPrefixAdaptiveNProfile(
        branches[0], branches[1], arena_bytes, arena_bytes + 1024
    )


def _resolve_m256_profile(
    group_count: int,
    dtype_bytes: int,
    output_dtype_bytes: int,
    capacity_bytes: int,
    native_a_offset: int,
) -> GroupedPrefixAdaptiveNProfile | None:
    """Pack each complete native M256 ring behind a common aligned header."""
    branches: list[GroupedPrefixNBranch] = []
    for physical_n, c_stages in ((224, 2), (256, 3)):
        for ab_stages in range(32, 1, -1):
            a_bytes = 256 * 64 * dtype_bytes * ab_stages // 2
            b_bytes = physical_n * 64 * dtype_bytes * ab_stages // 2
            d_bytes = 128 * 32 * output_dtype_bytes * c_stages
            packed_a = _align(d_bytes, 128)
            packed_b = _align(packed_a + a_bytes, 128)
            packed_end = packed_b + b_bytes
            native_b = _align(native_a_offset + a_bytes, 128)
            native_ab = _align(native_b + b_bytes, 8)
            native_d = _align(native_ab + 2 * ab_stages * 8, 1024)
            native_end = native_d + d_bytes
            # Include the SDK's initial 16-byte allocation offset. For all
            # admitted G<=32 the complete common state must fit before the
            # 1024-aligned payload, including the deeper branch's AB ring.
            header = _align(16 + 9 * 4, 16) + group_count * 16
            header = _align(header, 16) + group_count * 4
            header = _align(header, 4) + 4
            header = _align(header, 8) + 8 + 4 * 8 + 2 * 8
            header = _align(header, 8) + 2 * ab_stages * 8
            if (
                header <= 1024
                and packed_end + 1024 <= capacity_bytes
                and native_end <= capacity_bytes
            ):
                branches.append(
                    GroupedPrefixNBranch(
                        physical_n,
                        ab_stages,
                        c_stages,
                        a_bytes,
                        b_bytes,
                        d_bytes,
                        native_a_offset,
                        native_b,
                        native_ab,
                        native_d,
                        native_end,
                        packed_a,
                        packed_b,
                        0,
                    )
                )
                break
        else:
            return None
    arena = max(
        branch.packed_b_offset + branch.b_bytes
        for branch in branches
        if branch.packed_b_offset is not None
    )
    return GroupedPrefixAdaptiveNProfile(
        branches[0],
        branches[1],
        arena,
        arena + 1024,
        tmem_columns=512,
        common_d_tile=(128, 32),
        row_tile=256,
        layout_kind="operand_payload",
    )


def grouped_prefix_adaptive_n_analysis_eligible(
    analysis: Tcgen05GroupedWorklistAnalysis,
    *,
    matmul_operands_tma_provable: bool,
    source_m_tile: int = 128,
) -> bool:
    """Require the existing pure fresh-output proof and static input extents.

    First-binding integer hints remain seed-ranking inputs. This capability
    reads replayable tensor metadata and admits only actual static integers.
    """
    import torch

    left, right = analysis.packed_tensor, analysis.grouped_tensor
    if not (
        analysis.d_shift_supported
        and analysis.device_layout_kind == "offsets"
        and analysis.seed_facts.device_split_sizes
        and matmul_operands_tma_provable
        and left.ndim == 2
        and right.ndim == 3
        and left.is_contiguous()
        and right.is_contiguous()
        and left.dtype == right.dtype
        and left.dtype in (torch.float16, torch.bfloat16)
        and (analysis.output_dtype or left.dtype) in (torch.float16, torch.bfloat16)
        and analysis.metadata_tensor.dtype in (torch.int32, torch.int64)
    ):
        return False
    extents = (*left.shape, *right.shape)
    if not all(type(value) is int for value in extents):
        return False
    m, k = left.shape
    groups, n, rhs_k = right.shape
    return k == rhs_k and grouped_prefix_adaptive_n_input_domain(
        groups, m, n, k, source_m_tile=source_m_tile
    )
