"""Add a four-warp carrier for a pair of independent selected-row projections."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .....runtime.config import Config
from ..config_keys import STAGED_INDEXED_ROWS_KEY

if TYPE_CHECKING:
    from ....compile_environment import CompileEnvironment


# The registered carrier maps four row owners and a warp of K lanes onto
# the shared physical block. Other roots must fit this layout too.
STAGED_SEED_BLOCK = (4, 32, 1)


def add_indexed_row_seed(
    env: CompileEnvironment, configs: list[Config]
) -> list[Config]:
    spec = env.config_spec
    sites = spec.cute_staged_indexed_row_axes
    if not sites or not configs:
        return configs
    baxes = spec.block_sizes.valid_block_ids()
    taxes = spec.num_threads.valid_block_ids()
    for carrier in configs:
        opts = dict(carrier.config)
        bs = opts.get("block_sizes")
        nt = opts.get("num_threads")
        if (
            opts.get("cross_loop_pipeline") != "barrier"
            or not isinstance(bs, list)
            or not isinstance(nt, list)
            or len(bs) != len(baxes)
            or len(nt) != len(taxes)
            or not all(isinstance(x, int) and x > 0 for x in nt)
        ):
            continue
        blocks = list(bs)
        row_lanes, k_lanes, _ = STAGED_SEED_BLOCK
        counts = [min(row_lanes, value) for value in nt]
        seen: dict[int, tuple[int, int]] = {}
        has_paired = False
        # A compatible paired root already supplies 32 physical K lanes to
        # the shared CTA. Otherwise the unpaired root must supply its own.
        for row, kaxis, extent, end, paired in sorted(
            sites, key=lambda site: not site[-1]
        ):
            width = 16 if paired else 64
            # Four warps: paired rows use two K groups per output, and
            # adjacent rows use one warp per sixteen outputs.
            hardware_panel = 64 if paired else 32
            if (
                extent % width
                or end % hardware_panel
                or end // hardware_panel < 2
                or paired
                and end % 128
            ):
                continue
            update = {
                row: (width, row_lanes),
                kaxis: (
                    min(256 if paired else 64, end & -end),
                    k_lanes if paired or not has_paired else 2,
                ),
            }
            if any(
                a not in baxes or a not in taxes or a in seen and seen[a] != v
                for a, v in update.items()
            ):
                continue
            if any(
                not spec.block_sizes.block_id_lookup(a).min_size
                <= w
                <= spec.block_sizes.block_id_lookup(a).max_size
                for a, (w, _) in update.items()
            ):
                continue
            seen.update(update)
            has_paired |= paired
        if not seen:
            continue
        for a, (size, lanes) in seen.items():
            blocks[baxes.index(a)] = size
            counts[taxes.index(a)] = lanes
        opts.update(block_sizes=blocks, num_threads=counts, cute_defer_tile_reduce=True)
        # A strided multiwarp CTA cannot also implement the flat warp-row
        # schedule in this carrier. Leave the original seed untouched.
        opts.pop("cute_warp_row_reductions", None)
        opts[STAGED_INDEXED_ROWS_KEY] = True
        return [*configs, Config.from_dict(opts)]
    return configs
