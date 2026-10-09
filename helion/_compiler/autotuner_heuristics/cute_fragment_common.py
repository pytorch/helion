"""Root projection shared by computed-fragment algorithm capabilities."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...autotuner.compiler_coverage import CompilerCoverageGroup
from ...autotuner.compiler_coverage import CoverageWitness
from ...exc import InvalidConfig
from ...language import _tracing_ops
from ...runtime.config import Config

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ...autotuner.config_spec import ConfigSpec
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from ..device_ir import GraphInfo


def fragment_coverage_carrier(
    spec: ConfigSpec, resource_carrier: Config | None = None
) -> Config | None:
    """Strictly admit the carrier before enabling a new search coordinate."""
    generation = spec.create_config_generation()
    try:
        carrier = resource_carrier
        if carrier is None:
            _, carrier = generation.canonicalize_flat(generation.default_flat())
        generation.strict_config_pair(carrier)
    except InvalidConfig:
        return None
    return carrier


def register_fragment_boolean_coverage(
    spec: ConfigSpec, key: str, carrier: Config
) -> bool:
    """Append one independent Boolean witness after its field is enabled.

    Callers retain ownership of capability discovery, registration order and
    search-field rollback. Dependent witnesses keep their explicit declarations.
    """
    generation = spec.create_config_generation()
    try:
        generation.strict_config_pair(
            Config.from_dict(deepcopy(carrier.config) | {key: True})
        )
    except InvalidConfig:
        return False
    spec.register_compiler_coverage_group(
        CompilerCoverageGroup(
            mechanism=key.replace("cute_", "cute.", 1),
            version=1,
            key=key,
            domain=(False, True),
            legacy=False,
            witnesses=(CoverageWitness(carrier, True),),
            deferred=True,
        )
    )
    return True


@dataclass(frozen=True)
class FragmentRootRequirement:
    """Explicit Boolean capabilities required by one additional root."""

    root: int
    enabled_options: frozenset[str]


def active_fragment_roots(
    ordinary: frozenset[int],
    requirements: tuple[FragmentRootRequirement, ...],
    config: Mapping[str, object],
) -> frozenset[int]:
    """Select proved roots without changing or repairing the configuration."""
    return ordinary | frozenset(
        item.root
        for item in requirements
        if all(config.get(key) is True for key in item.enabled_options)
    )


def fragment_root_regions(ir: DeviceIR) -> list[tuple[int, list[GraphInfo]]]:
    """Project exactly the regions accepted by explicit ordered phase codegen."""
    from ..cute.materialized_fission_codegen import _region_graph_ids

    if len(ir.root_ids) == 1:
        return [(ir.root_ids[0], ir.graphs)]
    if not (
        ir.phases
        and not ir.implicit_dependency_starts
        and len(ir.phases) == len(ir.root_ids)
        and all(len(phase.roots) == 1 for phase in ir.phases)
        and all(
            node.target not in (_tracing_ops._if, _tracing_ops._while_loop)
            for graph in ir.graphs
            for node in graph.graph.nodes
        )
    ):
        return []
    result = []
    for root in ir.root_ids:
        graph_ids = _region_graph_ids(ir.graphs, root)
        result.append(
            (root, [graph for graph in ir.graphs if graph.graph_id in graph_ids])
        )
    return result


def computed_fragment_discovery_supported(
    env: CompileEnvironment, graphs: list[GraphInfo]
) -> bool:
    """Use live registered alias facts only for complete uniform-local discovery.

    Binding has not recorded the immutable storage facts when capability
    registration runs. Keep legacy admission first; only the existing uniform
    region proof may retry with its registered live disjointness classifier.
    Final code generation still requires the bound facts.
    """
    from ..cute.computed_fragment import computed_fragment_supported
    from ..cute.uniform_region_tree import uniform_local_regions

    if computed_fragment_supported(env, graphs):
        return True
    try:
        uniform_local_regions(graphs)
    except InvalidConfig:
        return False
    return computed_fragment_supported(env, graphs, allow_unbound=True)
