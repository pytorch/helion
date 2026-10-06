"""Orthogonal host-tail choices without replacing producer schedule seeds."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from ...runtime.config import Config
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .registry import CompilerHeuristicSpecializationFact


class CuteHostPairedSumHeuristic(AutotunerHeuristic):
    name = "cute_host_paired_sum"
    backend = "cute"

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        from ..cute.host_paired_sum import prove_host_sum_pairs
        from ..cute.host_single_sum import prove_host_single_sum

        host = device_ir.host_function
        env.config_spec.cute_host_paired_sum_available = bool(
            host is not None
            and (
                prove_host_sum_pairs(host).pairs
                or prove_host_single_sum(host).site is not None
            )
        )
        # Rank/dtype and constexpr values already participate in binding.
        # Shapes, strides, gradients and pointer alignment are runtime guards.
        return frozenset()

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return env.config_spec.cute_host_paired_sum_available


def add_host_sum_seeds(env: CompileEnvironment, seeds: list[Config]) -> list[Config]:
    """Place two independent tail choices early in the real initial population.

    Keep every existing seed in its original relative order, the promoted
    default unchanged, and the population budget unchanged. Only the primary
    producer receives siblings; copying all seeds would crowd out other classes.
    """
    from ..cute.host_paired_sum import PAIRED_SUM_KEY

    if not env.config_spec.cute_host_paired_sum_available:
        return seeds
    primary = seeds[0] if seeds else env.config_spec.default_config()
    choices = [
        Config.from_dict(deepcopy(primary.config) | {PAIRED_SUM_KEY: layout})
        for layout in ("mapped", "narrow")
    ]
    env.config_spec.autotuner_heuristics.append(CuteHostPairedSumHeuristic.name)
    return [*seeds[:1], *choices, *seeds[1:]]
