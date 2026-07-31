"""Auxiliary-tensor views of the generic tcgen05 epilogue plan."""

from __future__ import annotations

import dataclasses
from itertools import starmap
from typing import TYPE_CHECKING

import torch

from ...language import matmul_ops
from ...language import memory_ops
from ..compile_environment import CompileEnvironment
from .cute_epilogue import Tcgen05EpilogueLoadScope
from .cute_epilogue import Tcgen05EpiloguePlan
from .cute_fx_walk import build_inner_outputs_index_from_graphs
from .cute_fx_walk import reach_matmul_anchors

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST
    from ..host_function import HostFunction


_MMA_OPERAND_TRACE_THROUGH_TARGETS = (torch.ops.prims.convert_element_type.default,)
_TCGEN05_MMA_TARGETS = (
    torch.ops.aten.addmm.default,
    torch.ops.aten.mm.default,
    torch.ops.aten.bmm.default,
    torch.ops.aten.baddbmm.default,
    matmul_ops.dot,
)
_TCGEN05_NATIVE_MMA_OPERAND_DTYPES = (torch.bfloat16, torch.float16)


@dataclasses.dataclass(frozen=True)
class Tcgen05AuxTensorDescriptor:
    load_node: torch.fx.Node
    host_tensor_fx_node: torch.fx.Node
    host_tensor_val: torch.Tensor
    broadcast_axis: int | None
    store_value_node: torch.fx.Node


def aux_tensor_descriptors_from_epilogue_plan(
    plan: Tcgen05EpiloguePlan,
) -> tuple[Tcgen05AuxTensorDescriptor, ...]:
    """Adapt output-aligned load plans to the existing producer pipeline."""
    descriptors: list[Tcgen05AuxTensorDescriptor] = []
    seen_values: set[torch.fx.Node] = set()
    for store in plan.stores:
        if store.value_node in seen_values:
            continue
        seen_values.add(store.value_node)
        descriptors.extend(
            Tcgen05AuxTensorDescriptor(
                load_node=load.load_node,
                host_tensor_fx_node=load.host_tensor_fx_node,
                host_tensor_val=load.host_tensor_val,
                broadcast_axis=load.broadcast_axis,
                store_value_node=load.store_value_node,
            )
            for load in store.load_plans
            if load.scope is Tcgen05EpilogueLoadScope.OUTPUT_ALIGNED_SUBTILE
        )
    return tuple(descriptors)


def discover_tcgen05_aux_tensor_descriptors(
    cg: GenerateAST,
    matmul_fx_node: torch.fx.Node,
) -> tuple[Tcgen05AuxTensorDescriptor, ...]:
    plan = cg.device_function.cute_state.tcgen05_epilogue_plan_for_anchor(
        matmul_fx_node
    )
    if plan is None:
        return ()
    return aux_tensor_descriptors_from_epilogue_plan(plan)


def _trace_to_load_through_casts(
    node: torch.fx.Node,
) -> torch.fx.Node | None:
    current = node
    while current.op == "call_function" and current.target is not memory_ops.load:
        if current.target not in _MMA_OPERAND_TRACE_THROUGH_TARGETS:
            return None
        inputs = [arg for arg in current.args if isinstance(arg, torch.fx.Node)]
        if len(inputs) != 1:
            return None
        current = inputs[0]
    if current.op == "call_function" and current.target is memory_ops.load:
        return current
    return None


def _mma_facts(
    graphs: Sequence[GraphInfo],
) -> tuple[set[torch.fx.Node], set[torch.fx.Node]]:
    anchors: set[torch.fx.Node] = set()
    operand_loads: set[torch.fx.Node] = set()
    for graph_info in graphs:
        for node in graph_info.graph.nodes:
            if node.op != "call_function" or node.target not in _TCGEN05_MMA_TARGETS:
                continue
            anchors.add(node)
            for arg in node.args:
                if isinstance(arg, torch.fx.Node):
                    load = _trace_to_load_through_casts(arg)
                    if load is not None:
                        operand_loads.add(load)
    return anchors, operand_loads


def _store_pairs(
    graphs: Sequence[GraphInfo],
) -> tuple[tuple[torch.fx.Node, torch.fx.Node], ...]:
    result: list[tuple[torch.fx.Node, torch.fx.Node]] = []
    for graph_info in graphs:
        for node in graph_info.graph.nodes:
            if node.op != "call_function" or node.target is not memory_ops.store:
                continue
            value = node.args[2] if len(node.args) > 2 else None
            if isinstance(value, torch.fx.Node):
                result.append((node, value))
    return tuple(result)


def _reachable_non_operand_loads(
    graphs: Sequence[GraphInfo],
) -> tuple[tuple[torch.fx.Node, torch.fx.Node], ...]:
    anchors, operand_loads = _mma_facts(graphs)
    if not anchors:
        return ()
    inner_outputs = build_inner_outputs_index_from_graphs(graphs)
    result: list[tuple[torch.fx.Node, torch.fx.Node]] = []
    for store, value in _store_pairs(graphs):
        if not reach_matmul_anchors(
            value,
            target_fx_nodes=anchors,
            inner_outputs_by_graph_id=inner_outputs,
        ):
            continue
        visited: set[torch.fx.Node] = set()
        stack = [value]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            if (
                node.op == "call_function"
                and node.target is memory_ops.load
                and node not in operand_loads
            ):
                result.append((store, node))
            stack.extend(node.all_input_nodes)
    return tuple(result)


def host_function_has_tcgen05_aux_loads(
    host_function: HostFunction,
) -> bool:
    """Conservatively widen tuning when a kernel has an MMA and another load."""
    graphs = host_function.device_ir.graphs
    anchors, operand_loads = _mma_facts(graphs)
    return bool(anchors) and any(
        node.op == "call_function"
        and node.target is memory_ops.load
        and node not in operand_loads
        for graph_info in graphs
        for node in graph_info.graph.nodes
    )


def _has_same_tile_indices(load: torch.fx.Node, store: torch.fx.Node) -> bool:
    load_indices = load.args[1] if len(load.args) > 1 else None
    store_indices = store.args[1] if len(store.args) > 1 else None
    load_mask = load.args[2] if len(load.args) > 2 else None
    store_mask = store.args[3] if len(store.args) > 3 else None
    if (
        load_mask is not None
        or store_mask is not None
        or not isinstance(load_indices, (list, tuple))
        or not isinstance(store_indices, (list, tuple))
        or len(load_indices) != len(store_indices)
    ):
        return False
    return all(
        left is right
        or (
            isinstance(left, (int, slice, type(None)))
            and isinstance(right, type(left))
            and left == right
        )
        for left, right in zip(load_indices, store_indices, strict=True)
    )


def host_function_has_tcgen05_exact_shape_aux_loads(
    host_function: HostFunction,
) -> bool:
    """Widen the TMA-load axis for a same-shape, same-dtype external input."""
    compatible_stores: set[torch.fx.Node] = set()
    pairs = _reachable_non_operand_loads(host_function.device_ir.graphs)
    for store in {store for store, _ in pairs}:
        output_node = store.args[0] if store.args else None
        output = (
            output_node.meta.get("val")
            if isinstance(output_node, torch.fx.Node)
            else None
        )
        if not isinstance(output, torch.Tensor):
            continue
        exact_sources: list[torch.Tensor] = []
        for pair_store, load in pairs:
            if pair_store is not store:
                continue
            source_node = load.args[0] if load.args else None
            source = (
                source_node.meta.get("val")
                if isinstance(source_node, torch.fx.Node)
                else None
            )
            if isinstance(source, torch.Tensor) and _same_shape(source, output):
                if not _has_same_tile_indices(load, store):
                    return False
                exact_sources.append(source)
        if not exact_sources:
            continue
        if any(source.dtype != output.dtype for source in exact_sources):
            return False
        compatible_stores.add(store)
    return len(compatible_stores) == 1


def _same_shape(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.ndim != right.ndim:
        return False
    env = CompileEnvironment.current()
    return all(starmap(env.known_equal, zip(left.shape, right.shape, strict=True)))


def host_function_matmul_has_non_tcgen05_operand(
    host_function: HostFunction,
) -> bool:
    graphs = host_function.device_ir.graphs
    _, operand_loads = _mma_facts(graphs)
    return any(
        isinstance(value := load.meta.get("val"), torch.Tensor)
        and value.dtype not in _TCGEN05_NATIVE_MMA_OPERAND_DTYPES
        for load in operand_loads
    )
