"""Root projection shared by computed-fragment algorithm capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...exc import InvalidConfig
from ...language import _tracing_ops

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from ..device_ir import GraphInfo


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
