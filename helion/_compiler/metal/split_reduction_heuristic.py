"""The Metal autotuner's seeds for a few-row grid over a K pass that its
threads split and reduce after the loop.

A matrix-vector product written as a reduction (``acc[m, n, k_chunk] +=
x * w`` over a K loop, then ``acc.sum(-1)`` after it) has two grid tiles,
a few rows ``M`` of x and ``N`` output columns, and a pass block that cuts
K into chunks.  Every row of x reads every weight, so each tile of fewer
than ``M`` rows reads the whole weight matrix again: at 4 rows a 1-row tile
streams the weights 4 times.  The search's neighbours step one block or one
thread count at a time from the default, through layouts that read the
weights several times or leave the pass block on too few threads, and a
full search on an M4 Max settled on 1-row tiles at 0.37 ms for a 4-row,
6656 x 19968 4-bit matmul that runs in 0.19 ms (85% of DRAM bandwidth)
when every weight is read once.

Each seed puts the row block over all ``M`` rows, the pass block on one
SIMD group of 32 threads (whose reduction after the loop is a barrier-free
``simd_*`` collective) and 4 or 2 output columns per thread.  A thread
walks its rows and columns one after another around the K loop, so each
(row, column) pair runs a K pass of its own and re-reads its column's
weights; with every row in the threadgroup those re-reads come from cache,
and each weight comes from memory once instead of once per row tile.  On
the M4 Max at 4 rows, over the 4-, 5- and 6-bit matmuls of a 30B model's
verify step (``N x K`` from 256 x 6656 to 202048 x 6656), the 4-column
seed ran within 3% of the best of earlier tuned configs and a hand-written
kernel on every shape above 16 MB, the 2-column one on the 6-bit ones.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch._inductor.runtime.runtime_utils import next_power_of_2

from ...runtime.config import Config
from ..autotuner_heuristics.registry import AutotunerHeuristic
from ._constants import SIMD_WIDTH
from .backend import blocks_reduced_after_loop

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR

#: Most rows the row block covers whole, a K pass per row and column on
#: each thread: past them the grid has rows enough to fill the GPU without
#: reading each weight once, and the other seeds apply.  Measured at 4 rows;
#: up to here the seed is one candidate the search times.
MAX_ROWS = 32
#: Output columns per thread.
COLUMNS = (4, 2)


class MetalSplitReductionHeuristic(AutotunerHeuristic):
    name = "metal_split_reduction"
    backend = "metal"

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return bool(cls.get_seed_configs(env, device_ir))

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        seeds = cls.get_seed_configs(env, device_ir)
        return seeds[0] if seeds else None

    @classmethod
    def get_seed_configs(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> list[Config]:
        # ``device_ir`` imports ``helion.Config``, not yet defined when
        # ``runtime.kernel`` imports the heuristic registry.
        from ..device_ir import ForLoopGraphInfo

        spec = env.config_spec
        if spec.matmul_facts or spec.reduction_loops:
            return []
        loops = [i for i in device_ir.graphs if isinstance(i, ForLoopGraphInfo)]
        if len(loops) != 1 or len(loops[0].block_ids) != 1:
            return []
        (pass_block,) = loops[0].block_ids
        if len(device_ir.grid_block_ids) != 1 or len(device_ir.grid_block_ids[0]) != 2:
            return []
        grid = device_ir.grid_block_ids[0]
        blocks = {item.block_id: item for item in spec.block_sizes}
        if set(blocks) != {pass_block, *grid}:
            return []
        if set(spec.num_threads.valid_block_ids()) != set(blocks):
            return []
        if pass_block not in blocks_reduced_after_loop(device_ir):
            return []
        small = [b for b in grid if blocks[b].size_hint <= MAX_ROWS]
        if len(small) != 1:
            return []
        (rows_block,) = small
        (cols_block,) = (b for b in grid if b != rows_block)
        fragments = {b: blocks[b]._fragment(spec) for b in blocks}
        rows = next_power_of_2(blocks[rows_block].size_hint)
        seeds: list[Config] = []
        for columns in COLUMNS:
            chosen = {pass_block: SIMD_WIDTH, rows_block: rows, cols_block: columns}
            if not all(
                fragments[b].low <= size <= fragments[b].high
                for b, size in chosen.items()
            ):
                continue
            threads = {pass_block: SIMD_WIDTH, rows_block: 1, cols_block: 1}
            seed = Config(
                block_sizes=[chosen[i.block_id] for i in spec.block_sizes],
                num_threads=[threads[i.block_id] for i in spec.num_threads],
                num_warps=1,
            )
            seeds.append(seed)
        return seeds
