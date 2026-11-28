"""Geometry for staged paired BF16 row projections and their consumers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config_keys import STAGED_INDEXED_ROWS_KEY

if TYPE_CHECKING:
    from collections.abc import Iterator
    from collections.abc import Mapping

    from ....autotuner.config_spec import ConfigSpec
    from ....runtime.config import Config


def _stage_widths(
    spec: ConfigSpec, config: Mapping[str, object]
) -> Iterator[tuple[int, int]]:
    if (
        config.get(STAGED_INDEXED_ROWS_KEY) is not True
        or config.get("cross_loop_pipeline") != "barrier"
        or config.get("pid_type") != "persistent_interleaved"
        or config.get("cute_lane_unroll", 1) != 1
        or not isinstance(blocks := config.get("block_sizes"), list)
        or not isinstance(threads := config.get("num_threads"), list)
    ):
        return
    for row, k_axis, rows, k, paired in spec.cute_staged_indexed_row_axes:
        if not paired:
            continue
        width = (
            spec.block_sizes.config_get(blocks, row, 0)
            if row in spec.block_sizes.valid_block_ids()
            else spec.cute_tile_fixed_block_sizes.get(row)
        )
        x = spec.num_threads.config_get(threads, row, 0)
        y = spec.num_threads.config_get(threads, k_axis, 0)
        # Two warps cooperate for each eight-row slice. The emitter also
        # checks the actual physical CTA and the complete address recipe.
        if (
            type(width) is int
            and width > 0
            and width % 8 == 0
            and rows % width == 0
            and type(x) is int
            and type(y) is int
            and x > 0
            and y > 0
            and x * y == (width // 8) * 64
            and k % (x * y) == 0
            and k % 64 == 0
            and k >= 128
        ):
            yield row, width


def supports_row_epilogue(spec: ConfigSpec, config: Mapping[str, object]) -> bool:
    """The consumer must emit one scalar per logical tile, with a single writer."""
    if not spec.cute_indexed_row_epilogue_axes or not isinstance(
        blocks := config.get("block_sizes"), list
    ):
        return False
    staged = dict(_stage_widths(spec, config))
    tunable = spec.block_sizes.valid_block_ids()
    return any(
        producer in staged
        and all(
            (
                spec.block_sizes.config_get(blocks, axis, 0)
                if axis in tunable
                else spec.cute_tile_fixed_block_sizes.get(axis)
            )
            == 1
            for axis in consumer
        )
        for producer, *consumer in spec.cute_indexed_row_epilogue_axes
    )


def reusable_row_axes(spec: ConfigSpec, config: Mapping[str, object]) -> set[int]:
    """A persistent block must sometimes own a second tile of the same input."""
    if not spec.cute_reusable_row_inputs or config.get("num_sm_multiplier", 1) != 1:
        return set()
    return {
        row
        for row, width in _stage_widths(spec, config)
        if spec.cute_reusable_row_inputs.get(row, 0) // width > spec.num_sm > 0
    }


def pointwise_epilogue_carrier(
    spec: ConfigSpec, seed: Config
) -> dict[str, object] | None:
    """Give a compatible staged carrier one scalar lane per consumer tile."""
    if seed.get(STAGED_INDEXED_ROWS_KEY) is not True:
        return None
    if supports_row_epilogue(spec, seed):
        return seed.config.copy()
    baxes = spec.block_sizes.valid_block_ids()
    taxes = spec.num_threads.valid_block_ids()
    bs, ts = seed.get("block_sizes"), seed.get("num_threads")
    if (
        not isinstance(bs, list)
        or not isinstance(ts, list)
        or len(bs) != len(baxes)
        or len(ts) != len(taxes)
    ):
        return None
    for _, *consumer in spec.cute_indexed_row_epilogue_axes:
        # These axes belong to the component certified by the FX proof.
        blocks, threads = list(bs), list(ts)
        for axis in consumer:
            if axis in baxes:
                blocks[baxes.index(axis)] = 1
            if axis in taxes:
                threads[taxes.index(axis)] = 1
        candidate = seed.config | {
            "pid_type": "persistent_interleaved",
            "block_sizes": blocks,
            "num_threads": threads,
        }
        if supports_row_epilogue(spec, candidate):
            return candidate
    return None
