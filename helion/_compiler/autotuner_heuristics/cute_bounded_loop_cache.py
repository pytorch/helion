"""Coupled packet/cache seeds for equally bounded sibling row sweeps."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import torch

from ...runtime.config import Config
from .common import dedupe_configs
from .cute import _cute_tile_seed_vec_width_for_dtype
from .cute import _seq_config_list
from .cute_resident_sequence import CuteResidentSequenceHeuristic

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR


def _raw_load_vector_width(
    env: CompileEnvironment, device_ir: DeviceIR, columns: tuple[int, ...]
) -> int | None:
    """Choose a packet from loaded dtypes, before arithmetic promotion.

    Match the traced tile symbol, not a coincidentally equal size hint. Mixed
    input dtypes use the smallest common packet. This is only seed selection;
    generated address, mask and cache proofs still decide whether it is legal.
    """
    from ...language.memory_ops import load

    symbols = {env.block_sizes[block].symbol() for block in columns}
    widths = []
    for info in device_ir.graphs:
        for node in info.graph.nodes:
            value = node.meta.get("val")
            if (
                node.target is load
                and isinstance(value, torch.Tensor)
                and value.ndim > 0
                and isinstance(value.shape[-1], torch.SymInt)
                and value.shape[-1]._sympy_() in symbols
            ):
                widths.append(_cute_tile_seed_vec_width_for_dtype(value.dtype))
    if not widths or min(widths) == 1:
        return None
    return min(widths)


class CuteBoundedLoopCacheHeuristic(CuteResidentSequenceHeuristic):
    """Offer complete opt-in cache schedules without changing earlier seeds."""

    name = "cute_bounded_loop_cache"

    @classmethod
    def get_seed_configs(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> list[Config] | None:
        plan = cls._plan(env, device_ir)
        if plan is None:
            return None
        row, columns = plan
        vector = _raw_load_vector_width(env, device_ir, columns)
        if vector is None:
            return None
        spec = env.config_spec
        extent = env.block_sizes[columns[0]].size_hint()
        if extent <= 0:
            return None
        full_width = 1 << (extent - 1).bit_length()
        axes = {row, *columns}
        if any(
            len(item.block_ids) != 1 or item.block_id not in axes
            for item in spec.block_sizes
        ):
            return None
        seeds = []
        for rows, threads in (
            (1, 128),
            (1, 256),
            (1, 64),
            (1, 32),
            (2, 64),
            (4, 32),
            (2, 128),
            (4, 64),
        ):
            # Keep an owned lane loop in addition to the packet loop. Cache
            # indexing is proved from those lexical loops, not CUDA x/y hints.
            vector_limit = full_width // (2 * threads)
            if full_width // threads > 128 or vector_limit < 2:
                continue
            seed_vector = min(vector, vector_limit)
            sizes = {row: rows, **dict.fromkeys(columns, full_width)}
            if any(
                not item.min_size <= sizes[item.block_id] <= item.max_size
                for item in spec.block_sizes
            ):
                continue
            for schedule in ("bounded_layout", "bounded"):
                seeds.append(
                    Config(
                        block_sizes=cast(
                            "list[int]", _seq_config_list(spec.block_sizes, sizes)
                        ),
                        num_threads=cast(
                            "list[int]",
                            _seq_config_list(
                                spec.num_threads,
                                {
                                    row: rows if rows > 1 else 0,
                                    **dict.fromkeys(columns, threads),
                                },
                            ),
                        ),
                        cute_vector_widths=cast(
                            "list[int]",
                            _seq_config_list(
                                spec.cute_vector_widths,
                                {row: 1, **dict.fromkeys(columns, seed_vector)},
                            ),
                        ),
                        cute_lane_layouts=cast(
                            "list[str]",
                            _seq_config_list(
                                spec.cute_lane_layouts,
                                {row: "blocked", **dict.fromkeys(columns, "strided")},
                            ),
                        ),
                        cute_reduction_sequence=schedule,
                        cute_cluster_n=1,
                        pid_type="flat",
                    )
                )
        return dedupe_configs(seeds) or None
