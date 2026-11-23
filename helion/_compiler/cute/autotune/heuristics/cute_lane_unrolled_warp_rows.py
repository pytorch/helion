"""Several one-warp rows per CTA for the lane-unrolled row.

``CuteRolledRowLoadsFirstHeuristic`` seeds a row whose lanes issue their
loads first at one row per CTA.  Where one warp walks such a row in whole
lanes and the row stays persistent, a CTA of one warp leaves an SM at its
limit of 32 CTAs, 32 warps where its registers would hold more.  This
family seeds the same row at ``WARP_ROWS`` rows per CTA, one warp each,
with register reloads: the per-row code is the one-row CTA's and each warp
folds its own row.  On B200 a 16384 x 768 fp32 LayerNorm over the sum of a
contiguous operand and a stride-2 component read 0.990-0.997x of Inductor's
Triton kernel at four rows against 0.978-0.992x at one in the same
processes (a hand edit of the generated kernel that this layout renders),
and two rows read no gain.

The row stays persistent beside the wider tile axis because the config
names ``cute_lane_unroll`` (``_cute_config_forms_lane_unrolled_row``); a
kernel whose normalization rolls it anyway (one whose memory accesses may
need an order across threads, whose rows in registers the code generator
refuses), or drops the unroll, gets no seed.  Seeds only, appended after every older family
(``postprocess_seed_configs``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from .....exc import InvalidConfig
from .....runtime.config import Config
from ....autotuner_heuristics.registry import AutotunerHeuristic
from ...reduction_lanes import WARP_THREADS
from .cute import CuteRolledRowLoadsFirstHeuristic
from .cute import _cute_static_block_extent
from .cute import _is_sole_row_reduction
from .cute import _seq_config_list

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .....autotuner.config_spec import ConfigSpec
    from .....autotuner.config_spec import ReductionLoopSpec
    from ....compile_environment import CompileEnvironment
    from ....device_ir import DeviceIR

# Rows, one warp each, of a CTA.
WARP_ROWS = 4
# The lanes one thread keeps of its row: beyond it the row's values no
# longer stay in registers (the widest row a warp walks is 1024 elements).
MAX_THREAD_LANES = 32


class CuteLaneUnrolledWarpRowsHeuristic(AutotunerHeuristic):
    """``lane_unrolled_warp_rows_seed_configs``, appended after every other
    seed (``postprocess_seed_configs``); a seed family of that
    postprocessor, not an entry of the ranked registry."""

    name = "cute_lane_unrolled_warp_rows"
    backend = "cute"


def _lane_unrolled_row_seed(
    spec: ConfigSpec, configs: Sequence[Config], rdim_id: int
) -> Config | None:
    """The seed of ``configs`` that lays one lane-unrolled row per CTA with
    threads of its own (``CuteRolledRowLoadsFirstHeuristic``'s): the first
    that names ``cute_lane_unroll`` above 1, one row per block and a thread
    count for the reduction."""
    for seed in configs:
        unroll = seed.config.get("cute_lane_unroll", 1)
        threads = seed.config.get("num_threads")
        if (
            isinstance(unroll, int)
            and unroll > 1
            and seed.config.get("block_sizes") == [1]
            and isinstance(threads, list)
            and isinstance(
                count := spec.num_threads.config_get(threads, rdim_id, 0), int
            )
            and count > 0
        ):
            return seed
    return None


def lane_unrolled_warp_rows_seed_configs(
    env: CompileEnvironment, device_ir: DeviceIR, configs: Sequence[Config]
) -> list[Config]:
    """The lane-unrolled row seed of ``CuteRolledRowLoadsFirstHeuristic``
    among ``configs`` at ``WARP_ROWS`` rows per CTA, each on one warp,
    persistent, with register reloads; its unroll factor and lane vector are
    that seed's.

    No seed unless that heuristic seeded the kernel's rows (one tile axis
    beside the sole row reduction), the row extent is static, longer than a
    warp, a whole number of lanes of one (a multiple of 32, at most
    ``MAX_THREAD_LANES`` each) and the row axis takes ``WARP_ROWS`` rows per
    block; nor when the normalized seed is not that layout (the row rolled,
    as in a kernel that reads back storage it stores, or no lane loop that
    takes the unroll: ``ConfigSpec.cute_register_lane_rows``).
    """
    spec = env.config_spec
    if (
        CuteRolledRowLoadsFirstHeuristic.name not in spec.autotuner_heuristics
        or not _is_sole_row_reduction(env, device_ir)
        or len(spec.block_sizes) != 1
        or len(spec.reduction_loops) != 1
    ):
        return []
    rl_spec = cast("ReductionLoopSpec", spec.reduction_loops[0])
    rdim_id = rl_spec.block_id
    base = _lane_unrolled_row_seed(spec, configs, rdim_id)
    if base is None:
        return []
    row_spec = cast("Any", spec.block_sizes[0])
    extent = _cute_static_block_extent(env, rdim_id)
    if (
        extent is None
        or extent <= WARP_THREADS
        or extent % WARP_THREADS
        or extent // WARP_THREADS > MAX_THREAD_LANES
        or not row_spec.min_size <= WARP_ROWS <= row_spec.max_size
        or rdim_id not in spec.cute_reduction_reloads.valid_block_ids()
    ):
        return []
    values: dict[str, Any] = {
        **base.config,
        "block_sizes": [WARP_ROWS],
        "num_threads": _seq_config_list(
            spec.num_threads, {row_spec.block_id: WARP_ROWS, rdim_id: WARP_THREADS}
        ),
        "reduction_loops": [None],
        "cute_reduction_reloads": _seq_config_list(
            spec.cute_reduction_reloads, {rdim_id: "register"}
        ),
    }
    seed = Config.from_dict(values)
    try:
        normalized = spec.normalized_config(seed).config
    except InvalidConfig:
        return []
    if (
        normalized.get("reduction_loops") != [None]
        or normalized.get("cute_lane_unroll", 1) == 1
        or normalized.get("block_sizes") != [WARP_ROWS]
    ):
        return []
    return [seed]
