from __future__ import annotations

from typing import TYPE_CHECKING

from .....runtime.config import Config
from ....autotuner_heuristics.registry import AutotunerHeuristic
from ...warp_atomic import warp_atomic_candidate

if TYPE_CHECKING:
    from ....autotuner_heuristics.registry import CompilerHeuristicSpecializationFact
    from ....compile_environment import CompileEnvironment
    from ....device_ir import DeviceIR


class CuteWarpAtomicHeuristic(AutotunerHeuristic):
    """Offer transaction aggregation for contended integer ticket counters."""

    name = "cute_warp_atomic"
    backend = "cute"
    priority = 52

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        if any(
            warp_atomic_candidate(node)
            for info in device_ir.graphs
            for node in info.graph.nodes
        ):
            env.config_spec.cute_warp_aggregated_atomics_enabled = True
        return frozenset()

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return env.config_spec.cute_warp_aggregated_atomics_enabled

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        if not cls.is_eligible(env, device_ir):
            return None
        return Config(cute_warp_aggregated_atomics=True)
