"""Root projection shared by computed-fragment algorithm capabilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...language import _tracing_ops

if TYPE_CHECKING:
    from ..device_ir import DeviceIR
    from ..device_ir import GraphInfo


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
