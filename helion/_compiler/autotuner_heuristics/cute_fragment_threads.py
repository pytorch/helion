"""CTA worker-count control for proved complete computed-fragment roots."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from ...autotuner.compiler_coverage import CompilerCoverageGroup
from ...autotuner.compiler_coverage import CoverageWitness
from ...exc import InvalidConfig
from ...runtime.config import Config
from .cute_fragment_common import computed_fragment_discovery_supported
from .cute_fragment_common import fragment_coverage_carrier
from .cute_fragment_common import fragment_root_regions
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .registry import CompilerHeuristicSpecializationFact

KEY = "cute_fragment_threads"
# Existing coverage geometry is stable when the legal search domain grows.
LEGACY_COVERAGE_THREADS = (128, 32, 64, 256, 512)
THREADS = (*LEGACY_COVERAGE_THREADS, 1024)


class CuteFragmentThreadsHeuristic(AutotunerHeuristic):
    name = KEY
    backend = "cute"

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        host = device_ir.host_function
        assert host is not None
        with host:
            env.config_spec.cute_fragment_thread_root_ids = frozenset(
                root
                for root, graphs in fragment_root_regions(device_ir)
                if computed_fragment_discovery_supported(env, graphs)
            )
        # Shape/stride assumptions of an admitted root must survive rebinding.
        return (
            frozenset({"input_tensor_metadata"})
            if env.config_spec.cute_fragment_thread_root_ids
            else frozenset()
        )

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return False


def register_fragment_threads_coverage(
    env: CompileEnvironment,
    device_ir: DeviceIR,
    *,
    resource_carrier: Config | None = None,
) -> None:
    spec = env.config_spec
    if not spec.cute_fragment_thread_root_ids:
        return
    carrier = fragment_coverage_carrier(spec, resource_carrier)
    if carrier is None:
        return
    spec.cute_fragment_threads_search_enabled = True
    generation = spec.create_config_generation()
    try:
        for count in THREADS:
            requested = Config.from_dict(deepcopy(carrier.config) | {KEY: count})
            _, effective = generation.strict_config_pair(requested)
            if effective.config.get(KEY, 128) != count:
                spec.cute_fragment_threads_search_enabled = False
                return
    except InvalidConfig:
        spec.cute_fragment_threads_search_enabled = False
        return
    spec.register_compiler_coverage_group(
        CompilerCoverageGroup(
            mechanism="cute.computed_fragment_threads",
            version=1,
            key=KEY,
            domain=THREADS,
            legacy=128,
            # Keep the established four-witness budget. The appended 1024 value
            # remains an ordinary enum neighbor/random-search choice.
            witnesses=tuple(
                CoverageWitness(carrier, count) for count in LEGACY_COVERAGE_THREADS[1:]
            ),
            deferred=True,
        )
    )
