"""CTA worker-count control for proved complete computed-fragment roots."""

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

KEY = "cute_fragment_threads"
THREADS = (128, 32, 64, 256, 512)


class CuteFragmentThreadsHeuristic(AutotunerHeuristic):
    name = KEY
    backend = "cute"

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        from ..cute.computed_fragment import computed_fragment_supported

        host = device_ir.host_function
        assert host is not None
        with host:
            env.config_spec.cute_fragment_thread_root_ids = frozenset(
                root
                for root, graphs in fragment_root_regions(device_ir)
                if computed_fragment_supported(env, graphs)
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
    previous = spec.create_config_generation()
    try:
        if resource_carrier is None:
            _, carrier = previous.canonicalize_flat(previous.default_flat())
        else:
            carrier = resource_carrier
        previous.strict_config_pair(carrier)
    except InvalidConfig:
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
            witnesses=tuple(CoverageWitness(carrier, count) for count in THREADS[1:]),
            deferred=True,
        )
    )
