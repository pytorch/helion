"""Cooperative pointwise input publication for an original serial prefix.

Only a same-coordinate FP32 operand below the final multiply is materialized.
The scan buffer temporarily holds those inputs; the original serial consumer
reads slot i before publishing prefix i into it. No global boundary is changed.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from operator import itemgetter
from typing import TYPE_CHECKING

import torch
from torch.fx import Node

from ...language import _tracing_ops
from ...language import memory_ops
from .chained_pointwise_residency import _cost
from .chained_pointwise_residency import _pointwise
from .chained_pointwise_residency import _shape as resolved_shape
from .chained_root_native import joined_root
from .chained_root_native import native_facts
from .chained_store_expression import _node_facts
from .producer_phase import PointStore
from .producer_phase import PointValue

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST
    from .chained_execution import ChainedExecution
    from .chained_matmul import ChainedMatmulPlan
    from .chained_matmul import _Expression


KEY = "cute_chained_scan_input_materialize"


def _shape(node: Node) -> tuple[object, ...] | None:
    value = node.meta.get("val")
    return tuple(value.shape) if isinstance(value, torch.Tensor) else None


def _frontier(scan: Node) -> Node | None:
    """Keep the final multiply and every outer selection/cast on the prefix.

    These wrappers and pointwise broadcasting preserve the one input position.
    Views/gathers, alternate reductions and a materialized product are excluded.
    A nonlinear operation or selection is a boundary before that multiply, not
    a split of its possible fused multiply/add with the running prefix.
    """
    from .chained_collectives import classify_collective
    from .chained_matmul import _ancestors

    operation = classify_collective(scan)
    if operation is None or operation.kind != "scan" or operation.axis != 0:
        return None
    shape = _shape(operation.source)
    if shape is None or len(shape) != 1 or _shape(scan) != shape:
        return None
    node = operation.source
    while node.target in (
        torch.ops.aten.where.self,
        torch.ops.prims.convert_element_type.default,
        _tracing_ops._mask_to,
    ):
        if node.target is torch.ops.aten.where.self:
            values = [arg for arg in node.args[1:] if isinstance(arg, Node)]
            candidates = [arg for arg in values if _shape(arg) == shape]
            if len(candidates) != 1 or any(
                _shape(arg) not in ((), shape) for arg in node.all_input_nodes
            ):
                return None
            node = candidates[0]
        else:
            source = node.args[0]
            if not isinstance(source, Node) or _shape(source) != shape:
                return None
            if node.meta["val"].dtype != torch.float32:
                return None
            node = source
    if node.target is not torch.ops.aten.mul.Tensor or node.kwargs:
        return None
    choices = []
    for value in node.args:
        if (
            not isinstance(value, Node)
            or _shape(value) != shape
            or value.meta["val"].dtype != torch.float32
            or not _pointwise(value)
            or not (value.target is torch.ops.aten.where.self or _cost(value) >= 8)
        ):
            continue
        closure = _ancestors(value)
        cost = sum(_cost(item) for item in closure)
        if cost >= 8:
            choices.append((cost, value))
    return max(choices, key=itemgetter(0))[1] if choices else None


def has_scan_input_candidate(graphs: Sequence[GraphInfo]) -> bool:
    from .chained_matmul import _classify_chained_graph

    graph = _classify_chained_graph(graphs)
    # Rank-one scans without common-region reductions have a separate legacy
    # prelude. Do not advertise a choice that that emitter cannot consume.
    return (
        graph is not None
        and graph.loop is None
        and bool(graph.region.reductions)
        and any(_frontier(scan) is not None for scan in graph.scans)
    )


def _facts(plan: ChainedMatmulPlan) -> tuple[object, ...]:
    assert plan.region is not None
    return (
        plan.region,
        plan.dots,
        plan.axes,
        plan.threads,
        plan.strategy,
        native_facts(plan),
        plan.loop,
        plan.preparation_pipeline,
        plan.pointwise_cache,
        plan.root_vector_cache,
        plan.contraction_groups,
        plan.initialized_accumulator,
        plan.late_rhs_reuse,
        plan.root_typed_operands,
        plan.k_schedule,
        plan.warp_mma_stages,
        plan.warp_workspace,
        plan.warp_result_transport,
        plan.root_exports,
        plan.scan_exports,
        _node_facts(tuple(plan.region.graph.nodes)),
    )


@dataclass(frozen=True)
class ScanInput:
    scan: Node
    frontier: Node
    extent: int
    closure: tuple[Node, ...]
    facts: tuple[object, ...]

    def check(self, plan: ChainedMatmulPlan) -> None:
        from .chained_matmul import _ancestors
        from .chained_matmul import _UnsupportedChain

        if (
            self.facts != _facts(plan)
            or _frontier(self.scan) is not self.frontier
            or resolved_shape(self.scan) != (self.extent,)
            or set(self.closure) != _ancestors(self.frontier)
        ):
            raise _UnsupportedChain("scan input graph, coordinate or storage changed")


def plan_scan_inputs(plan: ChainedMatmulPlan) -> tuple[ScanInput, ...]:
    """Prove a raw-input prelude with no crossed effects or shared dependencies.

    This initial route uses the ordinary complete CTA and separately allocated
    collective buffers. Future loop/preparation/shared-input extensions need
    their own publication evidence; preinstalled names are not readiness.
    """
    from .chained_collectives import collective_nodes
    from .chained_collectives import uses_general_collectives
    from .chained_matmul import _ancestors
    from .chained_matmul import _supported

    region = plan.region
    if (
        region is None
        or not joined_root(plan)
        or plan.loop is not None
        or plan.preparation_pipeline is not None
        or plan.pointwise_cache is not None
        or plan.contraction_groups is not None
        or plan.initialized_accumulator is not None
        or plan.late_rhs_reuse is not None
        or plan.root_typed_operands is not None
        or plan.k_schedule is not None
        or plan.warp_mma_stages
        or not uses_general_collectives(plan)
    ):
        return ()
    positions = {node: index for index, node in enumerate(region.nodes)}
    materialized = {*plan.dots, *collective_nodes(plan)}
    result = []
    for scan in plan.scans:
        frontier = _frontier(scan)
        if frontier is None or positions[scan] >= positions[plan.dots[0]]:
            continue
        extent = resolved_shape(scan)[0]
        # The cooperative padded owner index is an Int32 expression.
        if type(extent) is not int or not 1 < extent <= 2**31 - plan.threads:
            continue
        prefix = region.nodes[: positions[scan] + 1]
        if any(
            node.op != "placeholder"
            and (
                node.op != "call_function"
                or node.target is memory_ops.store
                or node.is_impure()
                or not _supported(node)
            )
            for node in prefix
        ):
            continue
        closure = _ancestors(frontier)
        source = scan.args[1]
        assert isinstance(source, Node)
        source_closure = _ancestors(source)
        if materialized & source_closure:
            continue
        # Every replaced path is the direct pointwise image. Unknown views or
        # indexing between the frontier and prefix could reread an overwritten
        # slot; an exact fragment binding below also rejects such coordinates.
        consumers = source_closure - closure
        if any(
            frontier in _ancestors(node)
            and (
                not _pointwise(node)
                and node.target is not _tracing_ops._mask_to
                or _shape(node) != (extent,)
            )
            for node in consumers
        ):
            continue
        result.append(
            ScanInput(
                scan,
                frontier,
                extent,
                tuple(node for node in region.graph.nodes if node in closure),
                _facts(plan),
            )
        )
    return tuple(result)


def bind_scan_input(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    boundaries: dict[Node, str],
    selected: ScanInput,
    name: str,
    execution: ChainedExecution,
) -> tuple[PointValue, _Expression]:
    """Return typed producer plus a scan-local exact-coordinate substitution."""
    from .chained_collectives import collective_bindings
    from .chained_execution import ChainedExecution
    from .chained_matmul import _ancestors
    from .chained_matmul import _Expression
    from .chained_matmul import _UnsupportedChain

    selected.check(plan)
    scan_source = selected.scan.args[1]
    assert isinstance(scan_source, Node)
    if (
        execution != ChainedExecution(plan.threads)
        or boundaries.get(selected.scan) != name
        or collective_bindings(plan)[selected.scan] != name
        or any(node in boundaries for node in _ancestors(scan_source))
    ):
        raise _UnsupportedChain("scan input lacks its independent original buffer")
    position = f"{name}_input_position"
    source = _Expression(cg, plan, boundaries)
    source.coordinate_names.add(position)
    value = source.value(selected.frontier, (position,))
    point = PointValue(
        tuple(ast.parse("\n".join(source.lines)).body),
        ast.parse(f"cutlass.Float32({value})", mode="eval").body,
        PointStore(ast.parse(f"{name}[{position}]", mode="eval").body),
    )
    consumer = _Expression(cg, plan, boundaries)
    local = f"{name}_input_value"
    original_position = f"{name}_position"
    consumer.record_statement(
        ast.parse(f"{local} = {name}[{original_position}]").body[0]
    )
    consumer.fragments[selected.frontier] = ((original_position,), local)
    return point, consumer
