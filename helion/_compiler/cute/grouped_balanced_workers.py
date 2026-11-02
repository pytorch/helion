"""Device-only wave balancing for proved grouped persistent worklists."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


def balanced_workers_index_domain(
    groups: int, m: int, n: int, collective_m: int, collective_n: int
) -> bool:
    """Bound the SDK prefix sum even when every group spans the allocation.

    Device offsets are independent intervals. Their total can exceed the tile
    count of one packed allocation, so do not assume monotonic routing here.
    """
    limit = (1 << 31) - 1
    if (
        min(groups, m, n, collective_m, collective_n) <= 0
        or max(groups, m, n, collective_m, collective_n) > limit
        or m + collective_m - 1 > limit
        or n + collective_n - 1 > limit
    ):
        return False
    m_tiles = (m + collective_m - 1) // collective_m
    n_tiles = (n + collective_n - 1) // collective_n
    return groups * m_tiles * n_tiles <= limit


@dataclass(frozen=True)
class GroupedBalancedWorkersPlan:
    total_tiles: str
    workers: str
    grid: str
    collective_m: int
    collective_n: int
    cluster_m: int
    cluster_n: int = 1

    @property
    def active_predicate(self) -> str:
        # The total is a full-warp reduction of identically clipped per-CTA
        # metadata. All peers of a z cluster take the same structured branch.
        return (
            f"({self.total_tiles} > cutlass.Int32(0)) and "
            f"(cute.arch.block_idx()[2] < {self.workers})"
        )

    def setup_statements(
        self,
        problem_sizes: str,
        groups: int,
        new_var: Callable[[str], str],
    ) -> list[ast.stmt]:
        """Run after clipped SMEM metadata publication, before MMA setup.

        Every full warp reduces the same table. Consequently all warps and
        CTA peers choose the same count without another shared-memory barrier.
        The SDK uses this virtual grid for its persistent-loop stride. Its
        host-created params keep their original safe work-count upper bound.
        """
        group = new_var("tcgen05_balance_group")
        partial = new_var("tcgen05_balance_partial")
        rows = new_var("tcgen05_balance_rows")
        columns = new_var("tcgen05_balance_columns")
        maximum = new_var("tcgen05_balance_max_workers")
        waves = new_var("tcgen05_balance_waves")
        # Quotient/remainder ceil division avoids an Int32 addition overflow
        # when a proved total is close to the SDK's signed-coordinate limit.
        source = f"""
{partial} = cutlass.Int32(0)
for {group} in cutlass.range(cute.arch.lane_idx(), {groups}, 32, unroll=1):
    {rows} = {problem_sizes}[{group}, cutlass.Int32(0)]
    {columns} = {problem_sizes}[{group}, cutlass.Int32(1)]
    {partial} = {partial} + (({rows} + cutlass.Int32({self.collective_m - 1})) // cutlass.Int32({self.collective_m})) * (({columns} + cutlass.Int32({self.collective_n - 1})) // cutlass.Int32({self.collective_n}))
{self.total_tiles} = cute.arch.warp_reduction_sum({partial}, threads_in_group=32)
{maximum} = cutlass.Int32(cute.arch.grid_dim()[2])
{waves} = max(cutlass.Int32(1), {self.total_tiles} // {maximum} + cutlass.Int32({self.total_tiles} % {maximum} != cutlass.Int32(0)))
{self.workers} = max(cutlass.Int32(1), min({maximum}, {self.total_tiles} // {waves} + cutlass.Int32({self.total_tiles} % {waves} != cutlass.Int32(0))))
{self.grid} = (cutlass.Int32({self.cluster_m}), cutlass.Int32({self.cluster_n}), {self.workers})
"""
        return ast.parse(source).body
