"""Seed complete K tiles and a warp reduction for pure row accumulators."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

from .....runtime.config import Config
from ...row_carry_terms import can_unroll_carry
from ...warp_row_dot_spec import ROW_DOT_K
from ...warp_row_dot_spec import ROW_DOT_ROWS
from ...warp_row_dot_spec import row_dot_geometry
from ..config_keys import ROW_DOT_KEY

if TYPE_CHECKING:
    from .....autotuner.config_spec import ConfigSpec


def row_carry_seeds(spec: ConfigSpec) -> list[Config]:
    axes = spec.block_sizes.valid_block_ids()
    row_ids = spec.cute_fragment_row_carry_row_axes.intersection(axes)
    k_lengths: dict[int, list[int]] = {}
    for axis, extent in spec.cute_fragment_row_carry_contractions:
        k_lengths.setdefault(axis, []).append(extent)
    if not k_lengths or k_lengths.keys() - set(axes) or row_ids & k_lengths.keys():
        return []
    seeds = []
    seen = set()
    for rows, chunk in ((16, 256), (8, 128)):
        sizes: dict[int, int] = {}
        for axis, lengths in k_lengths.items():
            item = spec.block_sizes.block_id_lookup(axis)
            # Only complete power-of-two K tiles are supported by this lowering.
            upper = min(chunk, item.max_size, *(v & -v for v in lengths))
            if upper <= 0:
                break
            block = 1 << (upper.bit_length() - 1)
            if block < max(item.min_size, item.autotuner_min):
                break
            sizes[axis] = block
        else:
            for axis in row_ids:
                item = spec.block_sizes.block_id_lookup(axis)
                if item.max_size <= 0:
                    break
                upper = min(rows, item.max_size)
                block = max(
                    1 << (upper.bit_length() - 1), item.min_size, item.autotuner_min
                )
                if block > item.max_size:
                    break
                sizes[axis] = block
            else:
                # For the wider seed, coalesce short pointwise tiles in the
                # surrounding phases. The second seed keeps their reference layout.
                if rows == 16:
                    grid_ids = set(spec.grid_block_ids)
                    for item in spec.block_sizes:
                        if (
                            item.block_id not in sizes
                            and item.block_id in grid_ids
                            and item.size_hint >= 32
                        ):
                            ceiling = min(32, item.max_size)
                            candidate = 1 << (ceiling.bit_length() - 1)
                            if candidate >= max(item.min_size, item.autotuner_min):
                                sizes[item.block_id] = candidate
                # Other phase axes retain their ordinary reference tile sizes.
                defaults = cast(
                    "list[int]", spec.autotune_reference_config().config["block_sizes"]
                )
                blocks = [
                    sizes.get(item.block_id, default)
                    for item, default in zip(spec.block_sizes, defaults, strict=True)
                ]
                cfg = Config(block_sizes=blocks, cute_fragment_row_carry=True)
                if spec.cute_fragment_reduction_root_ids or (
                    spec.cute_fragment_row_carry_shared
                    and rows == 16
                    and any(
                        sizes.get(axis, 0) >= 32
                        for axis in spec.grid_block_ids
                        if axis not in k_lengths
                    )
                    and all(sizes[axis] >= 32 for axis in k_lengths)
                ):
                    # Keep a serial alternative when only the row-carry owner
                    # can use the warp option.
                    cfg.config["cute_fragment_reduction"] = "warp"
                if spec.cute_fragment_topk_network_root_ids:
                    cfg.config["cute_fragment_topk_network"] = True
                if spec.cute_fragment_row_carry_shared:
                    if (
                        spec.cross_loop_pipeline is None
                        or "barrier" not in spec.cross_loop_pipeline.choices
                        or "persistent_interleaved" not in spec.allowed_pid_types
                    ):
                        continue
                    # Flattening an omitted num_threads may assign threads to
                    # a K axis on a sibling root. The flat shared launch has
                    # no second dimension for those threads. Pin K even when
                    # an independent reduction owns a different warp.
                    cfg.config["num_threads"] = [
                        1
                        if axis in k_lengths
                        else (
                            0
                            if spec.cute_fragment_reduction_root_ids
                            else min(32, sizes.get(axis, 0))
                        )
                        for axis in spec.num_threads.valid_block_ids()
                    ]
                    cfg.config.update(
                        pid_type="persistent_interleaved",
                        cross_loop_pipeline="barrier",
                        num_sm_multiplier=1,
                    )
                key = (tuple(blocks), cfg.get("cute_fragment_reduction", "serial"))
                if key in seen:
                    continue
                seen.add(key)
                if cfg.get("cute_fragment_reduction") == "warp" and any(
                    sizes.get(axis) == tile
                    for axis, tile in spec.cute_fragment_row_carry_packet_tiles
                ):
                    packet = cfg.config | {"cute_fragment_vector_io": True}
                    unroll = any(
                        can_unroll_carry(extent // sizes[axis])
                        for axis, extent in spec.cute_fragment_row_carry_contractions
                    )
                    if unroll:
                        seeds.append(
                            Config.from_dict(
                                packet | {"cute_fragment_unrolled_loops": True}
                            )
                        )
                    seeds.append(Config.from_dict(packet))
                    if row_dot_geometry(spec, packet):
                        # Keep the smaller row tile as a control. An MMA warp
                        # computes sixteen output rows using the same packets.
                        # Widen only the row dimension owned by that warp.
                        dot = packet | {ROW_DOT_KEY: True}
                        if unroll:
                            dot["cute_fragment_unrolled_loops"] = True
                        seeds.append(Config.from_dict(dot))
                        dot_rows = {
                            row
                            for pairs in spec.cute_warp_row_dot_roots.values()
                            for row, k in pairs
                            if sizes.get(k) == ROW_DOT_K
                        }
                        # Widen the row tile up to eight 16-row MMA stripes.
                        # The preceding carrier still owns the thread geometry.
                        wide = [
                            max(block, min(8 * ROW_DOT_ROWS, item.max_size))
                            if item.block_id in dot_rows
                            and item.block_id not in k_lengths
                            else block
                            for item, block in zip(
                                spec.block_sizes, blocks, strict=True
                            )
                        ]
                        if wide != blocks:
                            seeds.append(Config.from_dict(dot | {"block_sizes": wide}))
                seeds.append(cfg)
    return seeds
