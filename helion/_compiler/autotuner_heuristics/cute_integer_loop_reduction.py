"""Deferred, default-off exact integer loop-reduction schedule."""

from __future__ import annotations

from contextlib import suppress
from copy import deepcopy
from typing import TYPE_CHECKING

from ...autotuner.compiler_coverage import CompilerCoverageGroup
from ...autotuner.compiler_coverage import CoverageWitness
from ...exc import InvalidConfig
from ...runtime.config import Config
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .registry import CompilerHeuristicSpecializationFact

KEY = "cute_integer_loop_reduction"


class CuteIntegerLoopReductionHeuristic(AutotunerHeuristic):
    name = KEY
    backend = "cute"

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        from ..cute.integer_loop_reduction import recurrence

        env.config_spec.cute_integer_loop_reduction_available = any(
            recurrence(info) is not None for info in device_ir.graphs
        )
        return frozenset()

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return False


def register_integer_loop_reduction_coverage(
    env: CompileEnvironment, device_ir: DeviceIR
) -> None:
    from ..cute.integer_loop_reduction import recurrence
    from ..device_ir import ForLoopGraphInfo

    spec = env.config_spec
    if not spec.cute_integer_loop_reduction_available:
        return
    previous = spec.create_config_generation()
    _, carrier = previous.canonicalize_flat(previous.default_flat())
    # Reuse ordinary tiling coordinates: one row per CTA and one complete
    # physical x tile for the recurrence. This is a candidate, not a substitute
    # for the configured ownership proof performed by BlockReductionStrategy.
    blocks = [
        info.block_ids[0]
        for info in device_ir.graphs
        if isinstance(info, ForLoopGraphInfo) and recurrence(info) is not None
    ]
    if len(blocks) == 1 and all(len(item.block_ids) == 1 for item in spec.block_sizes):
        block = blocks[0]
        requested = deepcopy(carrier.config)
        requested["block_sizes"] = [
            128 if item.block_id == block else 1 for item in spec.block_sizes
        ]
        requested["num_threads"] = [
            128 if item.block_id == block else 1 for item in spec.num_threads
        ]
        with suppress(InvalidConfig):
            _, carrier = previous.strict_config_pair(Config.from_dict(requested))
    spec.cute_integer_loop_reduction_search_enabled = True
    generation = spec.create_config_generation()
    try:
        generation.strict_config_pair(
            Config.from_dict(deepcopy(carrier.config) | {KEY: True})
        )
    except InvalidConfig:
        spec.cute_integer_loop_reduction_search_enabled = False
        return
    spec.register_compiler_coverage_group(
        CompilerCoverageGroup(
            mechanism="cute.integer_loop_reduction",
            version=1,
            key=KEY,
            domain=(False, True),
            legacy=False,
            witnesses=(CoverageWitness(carrier, True),),
            deferred=True,
        )
    )
