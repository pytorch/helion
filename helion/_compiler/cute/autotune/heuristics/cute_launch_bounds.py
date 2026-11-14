"""Additive coverage of the CuTe launch-bounds option (``cute_min_blocks_per_mp``):
a coverage group for matmul kernels, compiler seeds for flat pointwise ones."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from .....autotuner.compiler_coverage import CompilerCoverageGroup
from .....autotuner.compiler_coverage import CoverageWitness
from .....autotuner.compiler_coverage import same_value
from .....autotuner.config_fragment import EnumFragment
from .....exc import InvalidConfig
from .....runtime.config import Config
from ....autotuner_heuristics.registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ....autotuner_heuristics.registry import CompilerHeuristicSpecializationFact
    from ....compile_environment import CompileEnvironment
    from ....device_ir import DeviceIR


MIN_BLOCKS_KEY = "cute_min_blocks_per_mp"
MATMUL_MIN_BLOCKS_CHOICES = (0, 1)
# A pointwise kernel's tile at one element per thread in 128-thread CTAs, and
# the resident CTAs per SM its witness asks for: 10 caps the registers at 48
# (40 warps/SM), where better-benchmark's gsw_dHdT (pointwise_238e96847398),
# which ptxas leaves at 60 registers (8 CTAs), measured faster.
POINTWISE_THREADS = 128
POINTWISE_MIN_BLOCKS = 10
# The resident CTAs per SM ptxas reaches on its own at such a chain's
# registers: 8 at gsw_dHdT's 60, one kernel's measurement rather than a rule
# (a chain at 80 registers fits 6, one at 48 or fewer already fits 10).  A
# grid of no more CTAs than that per SM runs in one wave already, so the cap
# adds no warps; past it (1185 CTAs on a 148-SM B200), the cap's 10 per SM
# can fold a second wave into the first.
POINTWISE_RESIDENT_CTAS = 8


def register_matmul_min_blocks_coverage(
    env: CompileEnvironment, device_ir: DeviceIR
) -> None:
    """Pair one existing complete default with a one-block minimum hint.

    The automatic domain is deliberately narrower than the direct-config API.
    Neither normalization nor the launcher changes: previously supported
    explicit positive values remain available outside this automatic pair.
    """
    spec = env.config_spec
    if (
        env.backend_name != "cute"
        or not spec.matmul_facts
        or spec.cute_flash_search_enabled
        or spec.cute_scaled_mma_available
        or not spec.supports_config_key(MIN_BLOCKS_KEY)
        or MIN_BLOCKS_KEY in spec._flat_fields()
    ):
        return
    overrides = env.settings.autotune_config_overrides
    if MIN_BLOCKS_KEY in overrides and not any(
        same_value(overrides[MIN_BLOCKS_KEY], value)
        for value in MATMUL_MIN_BLOCKS_CHOICES
    ):
        # Global overrides apply to every coverage group. Keep the old schema
        # when this explicit request is outside our automatic domain, so an
        # unrelated group's strict transfer still accepts the existing option.
        # Seed-local requests do not constrain other groups or disable this pair.
        return
    try:
        previous = spec.create_config_generation()
        _, carrier = previous.canonicalize_flat(previous.default_flat())
        previous.strict_config_pair(carrier)
    except InvalidConfig:
        return
    if MIN_BLOCKS_KEY in carrier.config or any(
        not same_value(carrier.config.get(group.key, group.legacy), group.legacy)
        for group in spec.compiler_coverage_groups
    ):
        return

    # Expose the field and register its group together. A failed strict transfer
    # must leave the old field layout and any earlier coverage groups intact.
    spec.cute_matmul_min_blocks_search_enabled = True
    try:
        generation = spec.create_config_generation()
        for value in MATMUL_MIN_BLOCKS_CHOICES:
            requested = Config.from_dict(
                deepcopy(carrier.config) | {MIN_BLOCKS_KEY: value}
            )
            _, effective = generation.strict_config_pair(requested)
            if (
                not same_value(effective.config.get(MIN_BLOCKS_KEY), value)
                or _without_count(effective) != carrier.config
            ):
                spec.cute_matmul_min_blocks_search_enabled = False
                return
    except InvalidConfig:
        spec.cute_matmul_min_blocks_search_enabled = False
        return
    spec.register_compiler_coverage_group(
        CompilerCoverageGroup(
            mechanism="cute.matmul_min_blocks",
            version=1,
            key=MIN_BLOCKS_KEY,
            domain=MATMUL_MIN_BLOCKS_CHOICES,
            legacy=0,
            witnesses=tuple(
                CoverageWitness(carrier, value) for value in MATMUL_MIN_BLOCKS_CHOICES
            ),
        )
    )


def _pointwise_tile(env: CompileEnvironment) -> bool:
    """Whether the kernel's one loop is a one-dimensional root tile, with no
    reduction and no matmul, whose search offers ``POINTWISE_MIN_BLOCKS``:
    the kernels ``pointwise_min_blocks_seeds`` may seed, whatever the SM
    count and the tile's extent decide."""
    spec = env.config_spec
    fragment = spec._flat_fields().get(MIN_BLOCKS_KEY)
    paths = spec.cute_tile_loop_paths
    return (
        env.backend_name == "cute"
        and not spec.matmul_facts
        and not spec.reduction_block_ids
        and isinstance(fragment, EnumFragment)
        and POINTWISE_MIN_BLOCKS in fragment.choices
        and len(spec.block_sizes) == 1
        and len(spec.num_threads) == 1
        and len(paths) == 1
        and len(paths[0]) == 1
        and len(paths[0][0]) == 1
    )


class CutePointwiseMinBlocksHeuristic(AutotunerHeuristic):
    """``pointwise_min_blocks_seeds``, appended after every other seed
    (``postprocess_seed_configs``).  Whether they fire depends on the SM
    count (``ConfigSpec.num_sm``), so every kernel they may seed keys its
    binding on it, one below the cutoff included: the kernel binds again on
    a device with fewer SMs and gets a population of its own.  Under dynamic
    shapes the first binding's extent decides, as for every seed that reads
    a size hint; both seeds stay valid at any extent, so the binding keeps
    its shape buckets (``input_tensor_metadata`` would also make its hints
    proofs for code generation)."""

    name = "cute_pointwise_min_blocks"
    backend = "cute"
    CACHE_SPECIALIZATION_FACTS = frozenset({"config_num_sm"})

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        if not _pointwise_tile(env):
            return frozenset()
        return cls.CACHE_SPECIALIZATION_FACTS

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return bool(pointwise_min_blocks_seeds(env))


def pointwise_min_blocks_seeds(env: CompileEnvironment) -> list[Config]:
    """A pointwise kernel's default, retiled to one element per thread in
    ``POINTWISE_THREADS``-thread CTAs, at ``POINTWISE_MIN_BLOCKS`` resident
    CTAs per SM, and the same config at 0 (its control): compiler seeds, or
    none for another kernel.

    The search draws the count from its whole SIMT domain, but seldom beside
    the tile where the register cap pays off: a latency-bound chain at one
    element per thread.  The two seeds are measured like any other
    candidate, so a kernel that spills at the cap keeps its control, and
    the field stays in every random draw.  Only a kernel whose one loop is
    a one-dimensional root tile, with no reduction and no matmul, qualifies,
    and only while that tile spans more such CTAs than fit at
    ``POINTWISE_RESIDENT_CTAS`` per SM.
    """
    spec = env.config_spec
    if (
        not _pointwise_tile(env)
        or spec.block_sizes[0].size_hint
        <= POINTWISE_THREADS * POINTWISE_RESIDENT_CTAS * spec.num_sm
    ):
        return []
    generation = spec.create_config_generation()
    try:
        _, default = generation.canonicalize_flat(generation.default_flat())
        requested = deepcopy(default.config) | {
            "block_sizes": [POINTWISE_THREADS],
            "num_threads": [POINTWISE_THREADS],
        }
        _, carrier = generation.strict_config_pair(Config.from_dict(requested))
        _, witness = generation.strict_config_pair(
            Config.from_dict(
                deepcopy(carrier.config) | {MIN_BLOCKS_KEY: POINTWISE_MIN_BLOCKS}
            )
        )
    except InvalidConfig:
        return []
    if (
        carrier.config.get("block_sizes") != [POINTWISE_THREADS]
        or carrier.config.get("num_threads") != [POINTWISE_THREADS]
        or not same_value(carrier.config.get(MIN_BLOCKS_KEY, 0), 0)
        or not same_value(witness.config.get(MIN_BLOCKS_KEY), POINTWISE_MIN_BLOCKS)
        or _without_count(witness) != _without_count(carrier)
    ):
        return []
    return [witness, carrier]


def _without_count(config: Config) -> dict[str, object]:
    return {key: value for key, value in config.config.items() if key != MIN_BLOCKS_KEY}
