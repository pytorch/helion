"""MMA-DAG ownership of one explicit materialized phase."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

from ... import exc
from .materialized_fission_codegen import _region_graph_ids
from .materialized_fission_codegen import _stage_host
from .materialized_mma import _matmul_nodes
from .materialized_mma import _matmul_targets
from .ordered_phases import _ordered_phase_bundle

if TYPE_CHECKING:
    import ast

    from ..device_ir import DeviceIR
    from ..device_ir import GraphInfo
    from ..host_function import HostFunction


def project_ordered_mma_phase(
    device_ir: DeviceIR,
) -> tuple[HostFunction, list[GraphInfo], int, frozenset[int]] | None:
    """Project an ordered bundle with exactly one matmul-containing phase.

    The existing phase proof establishes explicit ordering and capture
    lifetime. Each sibling remains an ordinary matmul-free launch; the
    selected phase must still pass the complete MMA-DAG matcher.
    """
    host = device_ir.host_function
    if host is None or len(device_ir.phases) < 2:
        return None
    try:
        bundle, plan = _ordered_phase_bundle(host, device_ir)
    except exc.BackendUnsupported:
        return None
    regions = [
        _region_graph_ids(device_ir.graphs, root_id) for root_id in device_ir.root_ids
    ]
    targets = _matmul_targets()
    candidates = [
        index
        for index, region in enumerate(regions)
        if _matmul_nodes(
            [graph for graph in device_ir.graphs if graph.graph_id in region], targets
        )
    ]
    if len(candidates) != 1:
        return None
    (index,) = candidates
    root = cast("ast.For", bundle.body[plan.root_index + index])
    stage = _stage_host(bundle, root, index, device_ir.graphs, regions[index], plan)
    graphs = [
        graph for graph in stage.device_ir.graphs if graph.graph_id in regions[index]
    ]
    active = stage.device_ir.codegen_active_block_ids
    assert active is not None
    owned_blocks = set(active)
    for sibling_index, region in enumerate(regions):
        if sibling_index == index:
            continue
        sibling = _stage_host(
            bundle,
            cast("ast.For", bundle.body[plan.root_index + sibling_index]),
            sibling_index,
            device_ir.graphs,
            region,
            plan,
        )
        active = sibling.device_ir.codegen_active_block_ids
        assert active is not None
        owned_blocks.difference_update(active)
    return stage, graphs, index, frozenset(owned_blocks)
