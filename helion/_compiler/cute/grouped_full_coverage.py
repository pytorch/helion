"""Full-union metadata proof for a pure grouped contraction.

This helper consumes already normalized intervals. It never changes the source
width subtraction/clipping that creates them, or interprets offset tensor values
on the host. Admission of operands, stores and roles belongs to MMA lowering.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..ast_extension import statement_from_string

if TYPE_CHECKING:
    from ..device_function import DeviceFunction
    from .device_state import CuteTcgen05GroupedPlan


_INT32_MAX = (1 << 31) - 1
FULL_COVERAGE_PIPELINES = ((64, 2, 240), (128, 4, 256))


def full_coverage_pipeline_supported(
    block_k: object,
    ab_stages: object,
    consumer_regs: object,
    *,
    source_m_tile: object = 32,
) -> bool:
    """Complete ONE/source32 pipelines and the source64 K128 profile.

    Register values specify the consumer warp allocation, not the final ptxas
    register count. The latter is independently limited by the compiled object.
    """
    return all(
        type(value) is int
        for value in (block_k, ab_stages, consumer_regs, source_m_tile)
    ) and (
        (
            source_m_tile == 32
            and (block_k, ab_stages, consumer_regs) in FULL_COVERAGE_PIPELINES
        )
        or (
            source_m_tile == 64 and (block_k, ab_stages, consumer_regs) == (128, 4, 256)
        )
    )


def full_coverage_smem_upper_bound(
    groups: int, block_k: int, ab_stages: int, *, source_m_tile: int = 32
) -> int:
    """Bound every explicit allocation independent of compiler placement order.

    The older worklist estimator assumes source allocation order and charges
    only one AB barrier per stage. This profile has two barriers per AB stage.
    Charge each complete allocation plus its maximum alignment padding, including
    the final allocation, so reordered static shared symbols remain covered.
    Only the newly admitted K128 profile uses this stricter resource gate; the
    established K64 admission and its generated allocations stay unchanged.
    """
    assert type(groups) is int and groups > 0
    assert (block_k, ab_stages) in ((64, 2), (128, 4))
    assert type(source_m_tile) is int and source_m_tile in (32, 64)
    assert source_m_tile == 32 or (block_k, ab_stages) == (128, 4)
    allocations = (
        (9 * 4, 16),  # Scheduler mailbox.
        (groups * 4 * 4, 16),  # Device-derived problem sizes.
        (groups * 4, 16),  # Clipped starts.
        (4, 4),  # TMEM holding buffer.
        (8, 8),  # TMEM deallocation barrier.
        (2 * 2 * 8, 8),  # ACC2 full/empty barriers.
        (2 * 8, 8),  # One scheduler stage, full/empty barriers.
        (ab_stages * 128 * block_k * 2, 128),  # BF16 physical A stages.
        (ab_stages * source_m_tile * block_k * 2, 128),  # BF16 physical B stages.
        (2 * 128, 128),  # Original input descriptor workspace.
        (ab_stages * 2 * 8, 8),  # AB full/empty barriers.
        (128, 128),  # Original output descriptor workspace.
        (2 * 128 * 32 * 2, 1024),  # BF16 C2 store ring.
    )
    return sum(size + alignment - 1 for size, alignment in allocations)


def full_allocation_b_two_cta_profile_supported(
    block_m: object,
    block_n: object,
    block_k: object,
    ab_stages: object,
    consumer_regs: object,
    cluster_m: object,
    cluster_n: object,
) -> bool:
    """The complete TWO/source256 profile, independent of the ONE profiles."""
    values = (
        block_m,
        block_n,
        block_k,
        ab_stages,
        consumer_regs,
        cluster_m,
        cluster_n,
    )
    return all(type(value) is int for value in values) and values == (
        256,
        256,
        128,
        3,
        240,
        2,
        1,
    )


def full_allocation_b_two_cta_smem_upper_bound(groups: int) -> int:
    """Bound the TWO/AB3/ACC2/C2 allocations, including every alignment gap.

    Both operands have 256x128 elements per cluster and half that storage per
    CTA. Keep the original mutable A/B and D descriptor workspaces, paired
    TMEM deallocation state and both AB barriers per stage. The C2 ring uses
    the existing 128x32 epilogue subtile, not the full 256x256 output tile.
    """
    assert type(groups) is int and groups > 0
    allocations = (
        (9 * 4, 16),  # Scheduler mailbox, including all reserved slots.
        (groups * 4 * 4, 16),  # Device-derived problem sizes.
        (groups * 4, 16),  # Clipped starts.
        (4, 4),  # TMEM holding buffer.
        (8, 8),  # Paired-CTA TMEM deallocation barrier.
        (2 * 2 * 8, 8),  # ACC2 full/empty barriers.
        (2 * 8, 8),  # One scheduler stage, full/empty barriers.
        (3 * 128 * 128 * 2, 128),  # Physical A, per CTA.
        (3 * 128 * 128 * 2, 128),  # Physical B, per CTA.
        (2 * 128, 128),  # Original mutable input descriptor workspace.
        (3 * 2 * 8, 8),  # AB3 full/empty barriers.
        (128, 128),  # Original mutable output descriptor workspace.
        (2 * 128 * 32 * 2, 1024),  # BF16 C2 store ring.
    )
    return sum(size + alignment - 1 for size, alignment in allocations)


def full_coverage_index_domain(
    groups: int,
    m: int,
    n: int,
    k: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
) -> bool:
    """Bound all element products, tile coordinates and final loop increments.

    The ordinary device-metadata launcher bounds grid.z by
    G*ceil(M/tile_m)*ceil(N/tile_n), then caps it by persistent capacity.
    Retain that bound here, including the last dense Int32 loop increment.
    """
    values = (groups, m, n, k, tile_m, tile_n, tile_k)
    if any(type(value) is not int or not 0 < value <= _INT32_MAX for value in values):
        return False
    if m % tile_m or n % tile_n or k % tile_k:
        return False
    if m * k > _INT32_MAX or m * n > _INT32_MAX:
        return False
    if m + tile_m - 1 > _INT32_MAX or n + tile_n - 1 > _INT32_MAX:
        return False
    tiles = (m // tile_m) * (n // tile_n)
    grid_bound = groups * tiles
    return grid_bound <= _INT32_MAX and tiles - 1 + grid_bound <= _INT32_MAX


@dataclass(frozen=True)
class Tcgen05GroupedFullCoveragePlan:
    predicate: str
    groups: int
    m: int
    n: int
    k: int
    tile_m: int
    tile_n: int
    tile_k: int
    consumer_local: bool = False

    def __post_init__(self) -> None:
        assert self.predicate.isidentifier()
        assert type(self.consumer_local) is bool
        assert full_coverage_index_domain(
            self.groups, self.m, self.n, self.k, self.tile_m, self.tile_n, self.tile_k
        )

    @property
    def m_tiles(self) -> int:
        return self.m // self.tile_m

    @property
    def n_tiles(self) -> int:
        return self.n // self.tile_n

    @property
    def tiles(self) -> int:
        return self.m_tiles * self.n_tiles

    def tile_coordinates(self, index: str) -> tuple[str, str]:
        """The existing signed dense order, shared by publisher and consumers."""
        if self.m_tiles <= self.n_tiles:
            return (
                f"{index} % cutlass.Int32({self.m_tiles})",
                f"{index} // cutlass.Int32({self.m_tiles})",
            )
        return (
            f"{index} // cutlass.Int32({self.n_tiles})",
            f"{index} % cutlass.Int32({self.n_tiles})",
        )


def full_row_union_statements(
    df: DeviceFunction, grouped: CuteTcgen05GroupedPlan
) -> list[ast.stmt]:
    """Append to the original single-thread initializer, before its CTA fence.

    Every endpoint is normalized to [0,M]. A strict-progress chain needs at
    most G intervals; otherwise retain the complete original table byte for
    byte. A full union has one group spanning M and G-1 zero-length groups.
    """
    plan = grouped.full_coverage
    assert plan is not None
    assert grouped.clipped_negative_start and grouped.device_layout_kind == "offsets"
    assert grouped.m_size == plan.m and int(grouped.count) == plan.groups
    covered = df.new_var("tcgen05_row_union_covered")
    pass_index = df.new_var("tcgen05_row_union_pass")
    changed = df.new_var("tcgen05_row_union_changed")
    previous = df.new_var("tcgen05_row_union_previous")
    group = df.new_var("tcgen05_row_union_group")
    start = df.new_var("tcgen05_row_union_start")
    length = df.new_var("tcgen05_row_union_length")
    source = f"""
{covered} = cutlass.Int64(0)
{pass_index} = cutlass.Int32(0)
{changed} = cutlass.Boolean(True)
while {pass_index} < cutlass.Int32({plan.groups}) and {covered} < cutlass.Int64({plan.m}) and {changed}:
    {previous} = {covered}
    for {group} in range({plan.groups}):
        {start} = cutlass.Int64({grouped.starts}[cutlass.Int32({group})])
        {length} = cutlass.Int64({grouped.problem_sizes}[cutlass.Int32({group}), cutlass.Int32(1)])
        if {start} <= {covered}:
            {covered} = max({covered}, {start} + {length})
    {changed} = {covered} > {previous}
    {pass_index} = {pass_index} + cutlass.Int32(1)
if {covered} == cutlass.Int64({plan.m}):
    {grouped.starts}[cutlass.Int32(0)] = cutlass.Int32(0)
    {grouped.problem_sizes}[cutlass.Int32(0), cutlass.Int32(1)] = cutlass.Int32({plan.m})
    for {group} in range(1, {plan.groups}):
        {grouped.problem_sizes}[cutlass.Int32({group}), cutlass.Int32(1)] = cutlass.Int32(0)
"""
    return ast.parse(source).body


def full_row_union_predicate(grouped: CuteTcgen05GroupedPlan) -> ast.stmt:
    """Read the shared result only after the existing complete CTA barrier."""
    plan = grouped.full_coverage
    assert plan is not None
    return statement_from_string(
        f"{plan.predicate} = {grouped.starts}[cutlass.Int32(0)] == cutlass.Int32(0) "
        f"and {grouped.problem_sizes}[cutlass.Int32(0), cutlass.Int32(1)] "
        f"== cutlass.Int32({plan.m})"
    )
