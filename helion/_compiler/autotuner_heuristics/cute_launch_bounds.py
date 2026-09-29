"""Additive coverage of the existing CuTe matmul launch-bounds option."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from ...autotuner.compiler_coverage import CompilerCoverageGroup
from ...autotuner.compiler_coverage import CoverageWitness
from ...autotuner.compiler_coverage import same_value
from ...exc import InvalidConfig
from ...runtime.config import Config

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR


MIN_BLOCKS_KEY = "cute_min_blocks_per_mp"
MATMUL_MIN_BLOCKS_CHOICES = (0, 1)


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
                or {
                    key: setting
                    for key, setting in effective.config.items()
                    if key != MIN_BLOCKS_KEY
                }
                != carrier.config
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
