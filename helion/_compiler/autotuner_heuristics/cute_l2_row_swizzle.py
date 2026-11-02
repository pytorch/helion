"""Additive coverage for the persistent tcgen05 scheduler's L2 row swizzle.

``tcgen05_l2_swizzle_size`` permutes the tile -> CTA map of the persistent
tile scheduler (CUTLASS ``PersistentTileSchedulerParams``, raster along M):
with swizzle ``s`` the ``s`` consecutive clusters of the raster walk one
M row's N tiles. The first wave of clusters then shares each A row slab it
reads through L2 ``s`` ways and touches only ``ceil(clusters / s)`` of
them, which is what the kernel's cold-L2 start pays for; the default swizzle
of 1 gives every cluster of the wave its own row slab. The knob has always
been on the search surface; this group declares the row-covering value
(the deepest legal swizzle no larger than the output's N tile count) as a
witness on the complete carriers the other coverage groups already prove
-- the hoist carriers (fanout, pre-wait placement, hoist) where they exist,
else the fanout carriers, else the ordinary persistent seeds -- so a cold
search measures it against the same rings instead of waiting for a mutation
to land on it.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING
from typing import cast

from ...autotuner.compiler_coverage import CompilerCoverageGroup
from ...autotuner.compiler_coverage import CoverageDependency
from ...autotuner.compiler_coverage import CoverageWitness
from ...autotuner.config_fragment import EnumFragment
from ...exc import InvalidConfig
from ...runtime.config import Config
from ..cute.epilogue_fanout import FANOUT_CONFIG_KEY
from ..cute.epilogue_fanout import schedule_supported
from ..cute.strategies import TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY
from ..cute.tcgen05_constants import TCGEN05_AUX_FRAGMENT_HOIST_CONFIG_KEY

if TYPE_CHECKING:
    from ...autotuner.config_spec import ConfigSpec
    from ..compile_environment import CompileEnvironment
    from ..cute.tcgen05_config import CuteTcgen05Config
    from ..device_ir import DeviceIR

MECHANISM = "cute.l2_row_swizzle"
MAX_CARRIERS = 2


def row_swizzle_size(n_tiles: int, choices: tuple[int, ...]) -> int:
    """The deepest legal swizzle that does not exceed one row of N tiles."""
    legal = [value for value in choices if type(value) is int and value <= n_tiles]
    return max(legal) if legal else 1


def _carrier_sources(
    spec: ConfigSpec, state: CuteTcgen05Config
) -> tuple[list[Config], tuple[CoverageDependency, ...]]:
    """Complete carriers to witness on, with the groups they already depend on."""
    groups = {group.key: group for group in spec.compiler_coverage_groups}
    hoist = groups.get(TCGEN05_AUX_FRAGMENT_HOIST_CONFIG_KEY)
    if hoist is not None:
        sources: list[Config] = []
        for witness in hoist.witnesses:
            if witness.value is not True:
                continue
            carrier = Config.from_dict(
                deepcopy(witness.carrier.config)
                | {TCGEN05_AUX_FRAGMENT_HOIST_CONFIG_KEY: True}
            )
            if carrier not in sources:
                sources.append(carrier)
        dependencies = (
            *hoist.dependencies,
            CoverageDependency(
                hoist.mechanism, TCGEN05_AUX_FRAGMENT_HOIST_CONFIG_KEY, True
            ),
        )
        return sources, dependencies
    fanout = groups.get(FANOUT_CONFIG_KEY)
    if fanout is not None:
        sources = []
        for witness in fanout.witnesses:
            if witness.value != "shared":
                continue
            carrier = Config.from_dict(
                deepcopy(witness.carrier.config) | {FANOUT_CONFIG_KEY: "shared"}
            )
            if carrier not in sources:
                sources.append(carrier)
        return sources, (
            CoverageDependency(fanout.mechanism, FANOUT_CONFIG_KEY, "shared"),
        )
    sources = []
    for seed in spec.compiler_seed_configs:
        candidate = Config.from_dict(deepcopy(seed.config))
        try:
            spec.normalize(candidate)
        except InvalidConfig:
            continue
        if schedule_supported(candidate.config) and candidate not in sources:
            sources.append(candidate)
    return sources, ()


def register_l2_row_swizzle_coverage(
    env: CompileEnvironment, device_ir: DeviceIR
) -> None:
    """Declare the row-covering L2 swizzle on the complete persistent carriers."""
    spec = env.config_spec
    state = spec._cute_tcgen05_config
    if device_ir.host_function is None or not state.search_enabled:
        return
    fragment = spec._flat_fields().get(TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY)
    if (
        type(fragment) is not EnumFragment
        or fragment.search_choices is not None
        or fragment.coverage_choices is not None
        or len(fragment.choices) < 2
    ):
        return
    raw_choices = tuple(fragment.choices)
    legacy = fragment.default()
    if type(legacy) is not int or any(type(value) is not int for value in raw_choices):
        return
    choices = cast("tuple[int, ...]", raw_choices)
    extents = state.matmul_compile_time_static_extents
    indices = state._matmul_block_indices()
    if extents is None or indices is None:
        return
    _m, n, _k = extents
    if n is None or n <= 0:
        return
    n_index = indices[1]
    sources, dependencies = _carrier_sources(spec, state)
    witnesses: list[CoverageWitness] = []
    for source in sources:
        if len(witnesses) >= 2 * MAX_CARRIERS:
            break
        blocks = source.block_sizes
        if n_index >= len(blocks):
            continue
        bn = blocks[n_index]
        if type(bn) is not int or bn <= 0 or n % bn:
            continue
        row = row_swizzle_size(n // bn, choices)
        if row == legacy:
            continue
        carrier = Config.from_dict(deepcopy(source.config))
        carrier.config.pop(TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY, None)
        for value in (legacy, row):
            trial = Config.from_dict(
                deepcopy(carrier.config) | {TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY: value}
            )
            try:
                spec.normalize(trial)
            except InvalidConfig:
                break
            if trial.config.get(TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY) != value:
                break
        else:
            witnesses.extend(
                (CoverageWitness(carrier, legacy), CoverageWitness(carrier, row))
            )
    if not witnesses:
        return
    spec.register_compiler_coverage_group(
        CompilerCoverageGroup(
            mechanism=MECHANISM,
            version=1,
            key=TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY,
            domain=choices,
            legacy=legacy,
            witnesses=tuple(witnesses),
            dependencies=dependencies,
        )
    )
