"""Opt-in coverage for proved logical-coordinate complete-fragment gathers."""

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

KEY = "cute_fragment_bounded_gather"


def bounded_gather_roots(
    env: CompileEnvironment, device_ir: DeviceIR, *, allow_unbound: bool = False
) -> frozenset[int]:
    """The complete root proof, shared by explicit dependent capabilities.

    Discovery may use registered live alias facts before binding snapshots them.
    Code generation and other callers keep requiring immutable bound facts.
    """
    from ..cute.bounded_gather import prove_gather
    from ..cute.computed_fragment import computed_fragment_supported

    roots = set()
    host = device_ir.host_function
    assert host is not None
    with host:
        for root, graphs in fragment_root_regions(device_ir):
            if not any(
                prove_gather(env, node, graphs=graphs, allow_unbound=allow_unbound)
                is not None
                for info in graphs
                for node in info.graph.nodes
            ):
                continue
            if computed_fragment_supported(
                env, graphs, bounded_gather_owned=True, allow_unbound=allow_unbound
            ):
                roots.add(root)
    return frozenset(roots)


class CuteFragmentBoundedGatherHeuristic(AutotunerHeuristic):
    name = KEY
    backend = "cute"

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        roots = bounded_gather_roots(env, device_ir, allow_unbound=True)
        env.config_spec.cute_fragment_bounded_gather_root_ids = frozenset(roots)
        return frozenset({"input_tensor_metadata"}) if roots else frozenset()

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return False


def register_fragment_bounded_gather_coverage(
    env: CompileEnvironment,
    device_ir: DeviceIR,
    *,
    resource_carrier: Config | None = None,
) -> None:
    spec = env.config_spec
    if not spec.cute_fragment_bounded_gather_root_ids:
        return
    carrier = fragment_coverage_carrier(spec, resource_carrier)
    if carrier is None:
        return
    previous = spec.cute_fragment_bounded_gather_search_enabled
    spec.cute_fragment_bounded_gather_search_enabled = True
    if not register_fragment_boolean_coverage(spec, KEY, carrier):
        spec.cute_fragment_bounded_gather_search_enabled = previous
