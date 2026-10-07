"""Append-only coverage for costly pure producer materialization."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .cute_fragment_common import fragment_coverage_carrier
from .cute_fragment_common import fragment_root_regions
from .cute_fragment_common import register_fragment_boolean_coverage
from .cute_fragment_resources import fragment_resource_carrier
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ...runtime.config import Config
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .registry import CompilerHeuristicSpecializationFact

KEY = "cute_fragment_producer_cache"


class CuteFragmentProducerCacheHeuristic(AutotunerHeuristic):
    name = KEY
    backend = "cute"

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        from ..cute.computed_fragment import computed_fragment_supported
        from ..cute.producer_cache import producer_cache_candidates

        if not any(producer_cache_candidates(info.graph) for info in device_ir.graphs):
            return frozenset()
        host = device_ir.host_function
        assert host is not None
        with host:
            env.config_spec.cute_fragment_producer_cache_root_ids = frozenset(
                root
                for root, graphs in fragment_root_regions(device_ir)
                if computed_fragment_supported(env, graphs)
                and producer_cache_candidates(device_ir.graphs[root].graph)
            )
        return frozenset({"input_tensor_metadata"})

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return False


def register_fragment_producer_cache_coverage(
    env: CompileEnvironment,
    device_ir: DeviceIR,
    *,
    resource_carrier: Config | None = None,
) -> None:
    spec = env.config_spec
    if not spec.cute_fragment_producer_cache_root_ids:
        return
    # Preserve every old carrier/coverage witness. A cache-only root gets a
    # separate supplemental storage estimate, counting all lazy tensor outputs
    # as possible allocations just as the original no-reuse resource catalog.
    if not spec.cute_fragment_producer_cache_root_ids <= (
        spec.cute_fragment_scan_root_ids | spec.cute_fragment_reduction_root_ids
    ):
        resource_carrier = fragment_resource_carrier(
            env, device_ir, root_ids=spec.cute_fragment_producer_cache_root_ids
        )
    carrier = fragment_coverage_carrier(spec, resource_carrier)
    if carrier is None:
        return
    spec.cute_fragment_producer_cache_search_enabled = True
    register_fragment_boolean_coverage(spec, KEY, carrier)
