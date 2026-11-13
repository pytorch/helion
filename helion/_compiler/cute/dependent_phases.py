"""Prove two local dependent roots before assigning complete per-root owners."""

from __future__ import annotations

import ast
import dataclasses
from typing import TYPE_CHECKING

from ... import exc
from ...language import _tracing_ops
from ..compile_environment import CompileEnvironment
from .state import cute_env_state

if TYPE_CHECKING:
    from ...runtime.config import Config
    from ..device_ir import DeviceIR
    from ..device_ir import GraphInfo
    from ..host_function import HostFunction
    from ..tile_dependency import TileDependencyGraph


@dataclasses.dataclass(frozen=True)
class DependentPhasePlan:
    root_ids: tuple[int, ...]
    root_keys: tuple[str, ...]
    graph_regions: tuple[frozenset[int], ...]
    captures: tuple[str, ...]
    confirmed: bool = False


def _regions(ir: DeviceIR) -> tuple[frozenset[int], ...]:
    from ..device_ir import NodeArgsGraphInfo
    from .materialized_fission_codegen import _region_graph_ids

    if any(
        node.target is _tracing_ops._while_loop
        for info in ir.graphs
        for node in info.graph.nodes
    ):
        raise exc.BackendUnsupported("cute", "dependent phases exclude while state")
    result = tuple(
        _region_graph_ids(ir.graphs, root, allow_conditionals=True)
        for root in ir.root_ids
    )
    if result[0] & result[1]:
        raise exc.BackendUnsupported("cute", "dependent phases share a child graph")
    for region in result:
        graphs = {ir.graphs[index].graph for index in region}
        for index in region:
            info = ir.graphs[index]
            if isinstance(info, NodeArgsGraphInfo) and any(
                argument.graph not in graphs for argument in info.node_args
            ):
                raise exc.BackendUnsupported(
                    "cute", "dependent phase captures cross root graph ownership"
                )
    return result


def _local_boundary(graph: TileDependencyGraph) -> bool:
    return bool(graph.edges) and (
        {(edge.producer_root, edge.consumer_root) for edge in graph.edges} == {(0, 1)}
        and not graph.crosses_ranks()
        and not graph.inband_allocation_ids
    )


def prepare_required_owner_phases(ir: DeviceIR) -> bool:
    """Prove a per-root scope before reduction deferral; do not skip analysis."""
    from ..device_ir_analysis import DeviceIRAnalysis
    from ..tile_dependency import build_tile_dependency_graph
    from .ordered_phases import _capture_phase_bundle

    env = CompileEnvironment.current()
    state = cute_env_state(env)
    if state.dependent_phase_plan is not None:
        return tuple(ir.root_ids) == state.dependent_phase_plan.root_ids
    if not (
        len(ir.root_ids) == 2
        and len(ir.phases) == 1
        and ir.phases[0].roots == [0, 1]
        and not ir.implicit_dependency_starts
        and env.process_group_name is None
        and not env.backend.roots_fissioned(env)
    ):
        return False
    host = ir.host_function
    assert host is not None
    roots = ir.phases[0].root_nodes
    try:
        regions = _regions(ir)
        _bundle, capture = _capture_phase_bundle(host, ir, roots)
    except exc.BackendUnsupported:
        return False
    analysis = DeviceIRAnalysis.build(ir, env)
    graph = build_tile_dependency_graph(
        analysis.tile_accesses(ir, env, host),
        device_ir=ir,
        root_phases=(0, 0),
        world_size=1,
    )
    if not _local_boundary(graph):
        return False
    state.dependent_phase_plan = DependentPhasePlan(
        tuple(ir.root_ids),
        tuple(ast.dump(root) for root in roots),
        regions,
        capture.materialized_names,
    )
    return True


def confirm_dependency_phases(host: HostFunction, ir: DeviceIR) -> bool:
    """Recheck against the ordinary final analysis before choosing launches."""
    from .ordered_phases import _capture_phase_bundle

    state = cute_env_state(CompileEnvironment.current())
    plan = state.dependent_phase_plan
    if plan is None:
        return False
    graph = ir.tile_dependency_graph
    if not (
        tuple(ir.root_ids) == plan.root_ids
        and [phase.roots for phase in ir.phases] == [[0], [1]]
        and ir.implicit_dependency_starts == frozenset({1})
        and graph is not None
        and _local_boundary(graph)
    ):
        raise exc.BackendUnsupported(
            "cute", "dependent phase proof changed after lowering"
        )
    roots = [phase.root_nodes[0] for phase in ir.phases]
    _bundle, capture = _capture_phase_bundle(host, ir, roots)
    if (
        tuple(ast.dump(root) for root in roots) != plan.root_keys
        or capture.materialized_names != plan.captures
    ):
        raise exc.BackendUnsupported("cute", "dependent phase host capture changed")
    state.dependent_phase_plan = dataclasses.replace(
        plan, graph_regions=_regions(ir), confirmed=True
    )
    CompileEnvironment.current().config_spec.cute_dependent_phase_launches = True
    return True


def required_owner_regions(ir: DeviceIR) -> list[tuple[int, list[GraphInfo]]] | None:
    plan = cute_env_state(CompileEnvironment.current()).dependent_phase_plan
    if plan is None:
        return None
    if tuple(ir.root_ids) != plan.root_ids:
        raise exc.BackendUnsupported("cute", "dependent phase root identity changed")
    return [
        (root, [info for info in ir.graphs if info.graph_id in region])
        for root, region in zip(plan.root_ids, _regions(ir), strict=True)
    ]


def generate_dependent_phases(
    host: HostFunction, config: Config, emit_repro_caller: bool
) -> ast.Module:
    from .materialized_fission_codegen import generate_materialized_fission
    from .ordered_phases import _capture_phase_bundle

    plan = cute_env_state(CompileEnvironment.current()).dependent_phase_plan
    if plan is None or not plan.confirmed:
        raise exc.InvalidConfig("dependent phases require the final dependency proof")
    if config.pid_type != "flat" or any(
        config.get(key, False) for key in ("cute_pdl", "tcgen05_materialized_pdl")
    ):
        raise exc.InvalidConfig(
            "dependent phases require ordinary flat ordered launches"
        )
    confirm_dependency_phases(host, host.device_ir)
    bundle, capture = _capture_phase_bundle(
        host, host.device_ir, [phase.root_nodes[0] for phase in host.device_ir.phases]
    )
    return generate_materialized_fission(
        bundle, config, emit_repro_caller, capture, allow_conditionals=True
    )
