"""Deferred opt-in coverage for same-coordinate private atomic consumers."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from ...autotuner.compiler_coverage import CompilerCoverageGroup
from ...autotuner.compiler_coverage import CoverageWitness
from ...exc import InvalidConfig
from ...runtime.config import Config
from .cute_fragment_common import fragment_root_regions
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .registry import CompilerHeuristicSpecializationFact

KEY = "cute_fragment_atomic_consumer_fusion"


class CuteFragmentAtomicConsumerFusionHeuristic(AutotunerHeuristic):
    name = KEY
    backend = "cute"

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        from ..cute.atomic_consumer_fusion import atomic_consumer_regions
        from ..cute.computed_fragment import computed_fragment_supported

        host = device_ir.host_function
        assert host is not None
        roots = set()
        with host:
            for root, graphs in fragment_root_regions(device_ir):
                if computed_fragment_supported(env, graphs) and atomic_consumer_regions(
                    graphs, env
                ):
                    roots.add(root)
        env.config_spec.cute_fragment_atomic_consumer_fusion_root_ids = frozenset(roots)
        return frozenset({"input_tensor_metadata"}) if roots else frozenset()

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return False


def register_fragment_atomic_consumer_fusion_coverage(
    env: CompileEnvironment,
    device_ir: DeviceIR,
    *,
    resource_carrier: Config | None = None,
) -> None:
    spec = env.config_spec
    if not spec.cute_fragment_atomic_consumer_fusion_root_ids:
        return
    previous = spec.create_config_generation()
    try:
        carrier = resource_carrier
        if carrier is None:
            _, carrier = previous.canonicalize_flat(previous.default_flat())
        previous.strict_config_pair(carrier)
    except InvalidConfig:
        return
    previous_enabled = spec.cute_fragment_atomic_consumer_fusion_search_enabled
    spec.cute_fragment_atomic_consumer_fusion_search_enabled = True
    generation = spec.create_config_generation()
    try:
        generation.strict_config_pair(
            Config.from_dict(deepcopy(carrier.config) | {KEY: True})
        )
    except InvalidConfig:
        spec.cute_fragment_atomic_consumer_fusion_search_enabled = previous_enabled
        return
    spec.register_compiler_coverage_group(
        CompilerCoverageGroup(
            mechanism="cute.fragment_atomic_consumer_fusion",
            version=1,
            key=KEY,
            domain=(False, True),
            legacy=False,
            witnesses=(CoverageWitness(carrier, True),),
            deferred=True,
        )
    )
