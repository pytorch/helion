"""Seed for ``cute_tiles_per_block``: several one-element grid tiles per CTA.

A kernel whose grid axis is pinned to one element per tile (a GEMV: one row
per program, the K groups spread over the threads) launches one CTA per row.
The ``cute_tiles_per_block`` knob (``cute/tile_packing.py``) packs ``T`` such
tiles into one CTA as thread sub-groups; with a warp-wide K extent the layout
planner puts one warp on each tile and the K reduction is a single warp
shuffle.  The search finds the knob at small and medium row counts, but the
loads-first lane unroll (``cute_lane_unroll``) is only searched when a
compiler seed sets it, and the tall case (thousands of rows, no gain from
packing) wants exactly that: 32 threads over the whole K extent with every
lane's loads issued first.  This heuristic seeds both forms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...autotuner.config_spec import _CUTE_LANE_UNROLL_CHOICES
from ...runtime.config import Config
from ..cute.tile_packing import TILES_PER_BLOCK_CHOICES
from .cute import _seq_config_list
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR


class CuteTilePackWarpPerRowHeuristic(AutotunerHeuristic):
    """One warp per packed one-element grid tile, K lanes unrolled loads-first."""

    name = "cute_tile_pack_warp_per_row"
    backend = "cute"

    @classmethod
    def _plan(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> tuple[int, int, int, int, int] | None:
        """``(packed_block_id, tile_count, inner_block_id, inner_block, threads)``."""
        spec = env.config_spec
        if (
            spec.matmul_facts
            or spec.reduction_loops
            or not spec.cute_tile_pack_counts
            or len(spec.block_sizes) != 2
        ):
            return None
        packed_block_id, count = min(spec.cute_tile_pack_counts.items())
        inner = next(
            (item for item in spec.block_sizes if item.block_id != packed_block_id),
            None,
        )
        if inner is None or inner.block_id not in spec.num_threads.valid_block_ids():
            return None
        fragment = inner._fragment(spec)
        hint = max(1, inner.size_hint)
        # The whole inner extent per CTA (one trip of the inner loop), capped
        # by the fragment so the seed stays reachable.
        block = min(1 << (hint - 1).bit_length(), fragment.high)
        if block < fragment.low or block < 2:
            return None
        threads = min(32, block)
        return packed_block_id, count, inner.block_id, block, threads

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return cls._plan(env, device_ir) is not None

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
        plan = cls._plan(env, device_ir)
        if plan is None:
            return []
        packed_block_id, count, inner_block_id, block, threads = plan
        spec = env.config_spec
        lanes = block // threads
        unroll = lanes if lanes in _CUTE_LANE_UNROLL_CHOICES else 1
        packs = [
            pack
            for pack in sorted(TILES_PER_BLOCK_CHOICES, reverse=True)
            if pack > 1 and count % pack == 0
        ]
        seeds: list[Config] = []
        # The largest pack first, then the unpacked 32-thread form (tall
        # inputs), then the smaller packs.
        for pack in [*packs[:1], 1, *packs[1:]]:
            values: dict[str, object] = {
                "block_sizes": _seq_config_list(
                    spec.block_sizes, {packed_block_id: 1, inner_block_id: block}
                ),
                "num_threads": _seq_config_list(
                    spec.num_threads, {packed_block_id: 1, inner_block_id: threads}
                ),
                "cute_tiles_per_block": pack,
                "cute_lane_unroll": unroll,
            }
            if inner_block_id in spec.cute_lane_layouts.valid_block_ids():
                values["cute_lane_layouts"] = _seq_config_list(
                    spec.cute_lane_layouts, {inner_block_id: "strided"}
                )
            if inner_block_id in spec.cute_vector_widths.valid_block_ids():
                values["cute_vector_widths"] = _seq_config_list(
                    spec.cute_vector_widths, {inner_block_id: 1}
                )
            seeds.append(Config.from_dict(values))
        return seeds
