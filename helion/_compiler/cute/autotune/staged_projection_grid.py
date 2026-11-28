"""Bound opt-in cooperative grids by the logical tiles in staged projections."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ....exc import InvalidConfig
from ....runtime.config import Config
from .config_keys import PHASE_PARTICIPATION_KEY
from .config_keys import STAGED_GRID_KEY
from .config_keys import STAGED_INDEXED_ROWS_KEY

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ....autotuner.config_spec import ConfigSpec


def staged_task_bound(spec: ConfigSpec, config: Mapping[str, object]) -> int:
    """The largest static candidate task domain; check installed roots at rendering."""
    if (
        config.get(STAGED_INDEXED_ROWS_KEY) is not True
        or config.get("pid_type") != "persistent_interleaved"
        or config.get("cross_loop_pipeline") != "barrier"
        or config.get("cute_defer_tile_reduce") is not True
        or config.get("num_sm_multiplier", 1) != 1
        or config.get("cute_cluster_n", 1) != 1
        or spec.has_explicit_pdl
        or config.get("cute_pdl")
        or spec.num_sm <= 1
        or not isinstance(blocks := config.get("block_sizes"), list)
    ):
        return 0
    domain = 0
    tunable = spec.block_sizes.valid_block_ids()
    for axis, instances, rows in spec.cute_staged_row_domains:
        width = (
            spec.block_sizes.config_get(blocks, axis, 0)
            if axis in tunable
            else spec.cute_tile_fixed_block_sizes.get(axis)
        )
        if type(width) is int and width > 0 and rows % width == 0:
            domain = max(domain, instances * (rows // width))
    return domain if spec.num_sm < domain < 2**31 else 0


def normalize_staged_grid(
    spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    waves = config.get(STAGED_GRID_KEY, 1)
    active = config.get(PHASE_PARTICIPATION_KEY, False)
    if active is False:
        config.pop(PHASE_PARTICIPATION_KEY, None)
    elif (
        active is not True
        or type(waves) is not int
        or waves not in (2, 3)
        or not staged_task_bound(spec, config)
    ):
        if not fix_invalid:
            raise InvalidConfig("active phase joins require a checked staged grid")
        config.pop(PHASE_PARTICIPATION_KEY, None)
    if type(waves) is int:
        if waves == 1:
            config.pop(STAGED_GRID_KEY, None)
            return
        if waves in (2, 3) and staged_task_bound(spec, config):
            return
    if not fix_invalid:
        raise InvalidConfig(
            "staged projection grid requires two or three resident waves and a "
            "larger static domain on a single interleaved local barrier launch"
        )
    config.pop(STAGED_GRID_KEY, None)


def add_staged_grid_seeds(spec: ConfigSpec, configs: list[Config]) -> list[Config]:
    """Offer distinct bounded grids and a participating-CTA twin of the widest."""
    if not spec.cute_staged_row_domains:
        return configs
    for carrier in reversed(configs):
        bound = staged_task_bound(spec, carrier)
        if bound:
            widths: set[int] = set()
            added = []
            for waves in (2, 3):
                width = min(bound, waves * spec.num_sm)
                if width not in widths:
                    added.append(
                        Config.from_dict(carrier.config | {STAGED_GRID_KEY: waves})
                    )
                    widths.add(width)
            return [
                *configs,
                *added,
                Config.from_dict(added[-1].config | {PHASE_PARTICIPATION_KEY: True}),
            ]
    return configs
