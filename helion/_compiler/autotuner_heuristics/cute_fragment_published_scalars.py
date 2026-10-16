"""Additive coverage for published immutable scalar recipes."""

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

KEY = "cute_fragment_published_scalars"


class CuteFragmentPublishedScalarsHeuristic(AutotunerHeuristic):
    name = KEY
    backend = "cute"

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        from ..cute.computed_fragment import computed_fragment_supported
        from ..cute.published_scalars import scalar_source_candidates

        host = device_ir.host_function
        assert host is not None
        with host:
            env.config_spec.cute_fragment_published_scalar_root_ids = frozenset(
                root
                for root, graphs in fragment_root_regions(device_ir)
                if computed_fragment_supported(env, graphs)
                and scalar_source_candidates(graphs)
            )
        return (
            frozenset({"input_tensor_metadata"})
            if env.config_spec.cute_fragment_published_scalar_root_ids
            else frozenset()
        )

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return False


def register_fragment_published_scalars_coverage(
    env: CompileEnvironment,
    device_ir: DeviceIR,
    *,
    resource_carrier: Config | None = None,
) -> None:
    spec = env.config_spec
    if not spec.cute_fragment_published_scalar_root_ids:
        return
    carrier = fragment_coverage_carrier(spec, resource_carrier)
    if carrier is None:
        return
    spec.cute_fragment_published_scalars_search_enabled = True
    register_fragment_boolean_coverage(spec, KEY, carrier)
