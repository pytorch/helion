"""Offer complete-row publication on the staged multiwarp carrier."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .....runtime.config import Config
from ...loop_nesting import boundary_only_grid_blocks
from ...single_row_publication import row_pointwise_witness
from ...thread_budget import tile_loop_thread_count

if TYPE_CHECKING:
    from ....compile_environment import CompileEnvironment
    from ....device_ir import DeviceIR


def compatible_warp_rows(
    env: CompileEnvironment, ir: DeviceIR, seed: Config, warps: int
) -> bool:
    """The strided staged block can cover only serial chunks and complete row waves."""
    spec = env.config_spec
    facts = spec.cute_warp_row_reduction_roots
    blocks = seed.get("block_sizes")
    threads = seed.get("num_threads")
    chunks = seed.get("reduction_loops")
    if not (
        facts
        and isinstance(blocks, list)
        and isinstance(threads, list)
        and isinstance(chunks, list)
    ):
        return False
    for root, (row, k, extent, _count) in facts.items():
        rows = spec.block_sizes.config_get(blocks, row, 0)
        chunk = spec.reduction_loops.config_get(chunks, k)
        if not (
            isinstance(chunk, int)
            and chunk >= 32
            and chunk % 32 == 0
            and extent % chunk == 0
            and warps < rows <= 32
            and rows % warps == 0
            and spec.num_threads.config_get(threads, row, 0) == warps
            and spec.num_threads.config_get(threads, k, 0) == 1
            and row_pointwise_witness(ir.graphs[root].graph)
        ):
            return False
    return True


def add_publication_seeds(
    env: CompileEnvironment, ir: DeviceIR, configs: list[Config]
) -> list[Config]:
    """Retain the carrier and append only publication schedules it can cover."""
    facts = env.config_spec.cute_single_row_publication_roots
    if not facts or ir.host_function is None:
        return configs
    for seed in configs:
        if (
            not seed.get("cute_staged_indexed_rows")
            or seed.get("cross_loop_pipeline") != "barrier"
        ):
            continue
        if not isinstance(seed.get("block_sizes"), list) or not isinstance(
            seed.get("num_threads"), list
        ):
            continue
        # Use the seed's tile nesting and thread counts, including axes
        # supplied by other roots. The final host launch remains authoritative.
        with ir.host_function:
            inactive = boundary_only_grid_blocks(env, ir, ir.graphs)
            threads = tile_loop_thread_count(
                env, ir, ir.graphs, seed, inactive_block_ids=inactive
            )
        if not (
            32 <= threads <= 1024
            and threads % 32 == 0
            and any(width % threads == 0 for _, width in facts.values())
        ):
            continue
        # The interleaved carrier exposes the proved roots to the staged emitter.
        options = seed.config | {
            "pid_type": "persistent_interleaved",
            "cute_single_row_publication": True,
        }
        sibling = compatible_warp_rows(env, ir, seed, threads // 32)
        combined = options | {"cute_warp_row_reductions": True} if sibling else options
        twins = [
            Config.from_dict(combined),
            Config.from_dict(combined | {"cute_parallel_row_sum": True}),
        ]
        if sibling:
            # Keep the serial writer independent of the sibling reduction.
            twins.append(Config.from_dict(options))
        return [*configs, *twins]
    return configs
