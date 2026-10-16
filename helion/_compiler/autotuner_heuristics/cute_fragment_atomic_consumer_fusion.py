"""Deferred opt-in coverage for same-coordinate private atomic consumers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .cute_fragment_common import fragment_coverage_carrier
from .cute_fragment_common import fragment_root_regions
from .cute_fragment_common import register_fragment_boolean_coverage
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ...runtime.config import Config
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
    extended_only: bool = False,
) -> None:
    spec = env.config_spec
    if not spec.cute_fragment_atomic_consumer_fusion_root_ids:
        return
    from ..cute.atomic_consumer_fusion import _local_regions

    host = device_ir.host_function
    assert host is not None
    with host:
        legacy = any(
            root in spec.cute_fragment_atomic_consumer_fusion_root_ids
            and _local_regions(graphs, env)
            for root, graphs in fragment_root_regions(device_ir)
        )
    # Keep every old eligible kernel's witness at its historical position.
    # Newly eligible owner/store regions append after all prior coverage groups.
    if extended_only == bool(legacy):
        return
    carrier = fragment_coverage_carrier(spec, resource_carrier)
    if carrier is None:
        return
    previous_enabled = spec.cute_fragment_atomic_consumer_fusion_search_enabled
    spec.cute_fragment_atomic_consumer_fusion_search_enabled = True
    if not register_fragment_boolean_coverage(spec, KEY, carrier):
        spec.cute_fragment_atomic_consumer_fusion_search_enabled = previous_enabled
