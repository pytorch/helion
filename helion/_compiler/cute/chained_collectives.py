"""Materialized scans and reductions inside a computed-contraction region.

The expression emitter supplies the source arithmetic at logical coordinates.
Only ownership and communication are selected here: one warp per reduction
vector, or a lane/warp per prefix. Results stay FP32 and can be consumed by
any later contraction, pointwise producer or store in the region.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING
from typing import Literal

import torch
from torch.fx import Node

from ...language import scan_ops
from .chained_execution import ChainedExecution
from .producer_phase import CollectiveProducer
from .producer_phase import collective_body
from .producer_phase import emit_producer_phase

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST
    from .chained_matmul import ChainedMatmulPlan
    from .chained_scratch_layout import ScratchLayouts


@dataclass(frozen=True)
class Collective:
    node: Node
    source: Node
    axis: int
    kind: str


def classify_collective(node: Node) -> Collective | None:
    """Classify the source operation; scan combiner validation is separate."""
    if node.target is scan_ops._associative_scan:
        if len(node.args) != 5 or node.args[3:] != (False, False) or node.kwargs:
            return None
        source, axis = node.args[1:3]
        kind = "scan"
    elif node.target is torch.ops.aten.sum.dim_IntList:
        if not 2 <= len(node.args) <= 3 or set(node.kwargs) - {"dtype", "keepdim"}:
            return None
        source, axes = node.args[:2]
        if not isinstance(axes, (tuple, list)) or len(axes) != 1:
            return None
        axis = axes[0]
        kind = "sum"
    else:
        return None
    if not isinstance(source, Node) or type(axis) is not int:
        return None
    value, result = source.meta.get("val"), node.meta.get("val")
    if (
        not isinstance(value, torch.Tensor)
        or not isinstance(result, torch.Tensor)
        or value.dtype != torch.float32
        or result.dtype != torch.float32
        or value.ndim not in (1, 2)
        or not -value.ndim <= axis < value.ndim
        or (kind == "sum" and value.ndim != 2)
    ):
        return None
    return Collective(node, source, axis % value.ndim, kind)


def uses_general_collectives(plan: ChainedMatmulPlan) -> bool:
    from .chained_matmul import _ancestors

    return plan.region is not None and (
        bool(plan.region.reductions)
        or plan.loop is not None
        and bool(plan.region.scans)
        or any(node.meta["val"].ndim != 1 for node in plan.region.scans)
        # The legacy rank-one prelude runs before every contraction. A scan
        # that reads an MMA result must instead run at its graph position.
        or any(
            set(plan.dots) & _ancestors(scan.args[1])
            for scan in plan.region.scans
            if isinstance(scan.args[1], Node)
        )
    )


def has_parallel_scan_candidate(graphs: Sequence[GraphInfo]) -> bool:
    """Discover nontrivial prefixes on the common graph-positioned path."""
    from ..compile_environment import CompileEnvironment
    from ..compile_environment import FixedBlockSizeSource
    from .chained_matmul import _ancestors
    from .chained_matmul import _classify_chained_graph

    graph = _classify_chained_graph(graphs)
    if graph is None:
        return False
    env = CompileEnvironment.current()
    general = (
        graph.loop is not None
        or bool(graph.region.reductions)
        or any(node.meta["val"].ndim != 1 for node in graph.scans)
        or any(set(graph.dots) & _ancestors(node) for node in graph.scans)
    )
    if not general:
        return False
    for node in graph.scans:
        operation = classify_collective(node)
        if operation is None:
            continue
        size = operation.source.meta["val"].shape[operation.axis]
        if isinstance(size, int):
            if size > 1:
                return True
            continue
        block_id = env.get_block_id(size)
        if block_id is None:
            continue
        block_id = env.canonical_block_id(block_id)
        source = env.block_sizes[block_id].block_size_source
        if isinstance(source, FixedBlockSizeSource):
            if type(source.value) is int and source.value > 1:
                return True
        elif block_id in env.config_spec.block_sizes.valid_block_ids():
            if env.config_spec.block_sizes.block_id_lookup(block_id).max_size > 1:
                return True
    return False


def validate_scan_schedule(plan: ChainedMatmulPlan | None, mode: object) -> None:
    """Reject choices that would silently disappear in a different emitter."""
    from ... import exc
    from .chained_matmul import _shape

    if mode == "serial":
        return
    if (
        mode != "warp"
        or plan is None
        or not uses_general_collectives(plan)
        or not any(
            operation is not None and _shape(operation.source)[operation.axis] > 1
            for node in plan.scans
            for operation in (classify_collective(node),)
        )
    ):
        raise exc.BackendUnsupported(
            "cute", "warp scan requires a nontrivial common-region prefix"
        )


def warp_prefix_point(
    name: str,
    extent: int,
    expression: list[str],
    value: str,
    *,
    execution: ChainedExecution,
) -> list[str]:
    """The original ordered prefix for one warp/part, without publication.

    The caller binds ``name_part`` and the source coordinates and executes all
    lanes of a warp. This helper changes neither the initial addition nor the
    five shuffle/add steps. It grants no storage, lifetime or mask authority.
    """
    from .chained_matmul import _indent

    position = f"{name}_position"
    acc, carry = f"{name}_acc", f"{name}_carry"
    body = [
        f"{position} = {execution.thread} % 32 + {name}_part * 32",
        f"{acc} = cutlass.Float32(0)",
        f"if {position} < {extent}:",
        _indent(expression),
        f"    {acc} = cutlass.Float32(0) + {value}",
    ]
    for offset in (1, 2, 4, 8, 16):
        other = f"{name}_up_{offset}"
        body.extend(
            [
                f"{other} = cute.arch.shuffle_sync_up({acc}, offset={offset})",
                f"if {execution.thread} % 32 >= {offset}:",
                f"    {acc} = {other} + {acc}",
            ]
        )
    if extent > 32:
        body.extend(
            [
                f"if {name}_part > 0:",
                f"    {acc} = {carry} + {acc}",
                f"{carry} = cute.arch.shuffle_sync({acc}, 31)",
            ]
        )
    return body


def _warp_prefix(
    name: str,
    extent: int,
    vectors: int,
    threads: int,
    coords: tuple[str, ...],
    expression: list[str],
    value: str,
    *,
    execution: ChainedExecution | None = None,
) -> list[str]:
    """Compatibility wrapper for the shared original prefix recipe."""
    execution = execution or ChainedExecution(threads)
    return collective_body(
        CollectiveProducer(
            name,
            "warp_scan",
            extent,
            vectors,
            coords,
            coords,
            tuple(expression),
            value,
            execution,
        )
    )


def collective_nodes(plan: ChainedMatmulPlan) -> tuple[Node, ...]:
    assert plan.region is not None
    selected = set(plan.region.scans) | set(plan.region.reductions)
    return tuple(node for node in plan.region.nodes if node in selected)


def collective_bindings(plan: ChainedMatmulPlan) -> dict[Node, str]:
    return {
        node: f"chain_collective_{index}"
        for index, node in enumerate(collective_nodes(plan))
    }


def needs_materialized_final(plan: ChainedMatmulPlan) -> bool:
    """A collective may redistribute the final MMA result across threads."""
    from .chained_matmul import _ancestors

    return any(plan.dots[-1] in _ancestors(node) for node in collective_nodes(plan))


def workspace_bytes(plan: ChainedMatmulPlan) -> int:
    from .chained_matmul import _shape

    return sum(
        (4 * math.prod(_shape(node)) + 127) // 128 * 128
        for node in collective_nodes(plan)
    )


def allocate_collectives(plan: ChainedMatmulPlan, scratch: ScratchLayouts) -> list[str]:
    from .chained_matmul import _shape

    return [
        f"{name} = cute.make_tensor(cute.arch.alloc_smem(cutlass.Float32, {math.prod(_shape(node))}, alignment=128), {scratch.layout(name, _shape(node))})"
        for node, name in collective_bindings(plan).items()
    ]


def _coordinates(axis: int, vector: str, position: str, rank: int) -> tuple[str, ...]:
    if rank == 1:
        return (position,)
    return (position, vector) if axis == 0 else (vector, position)


def emit_collectives_before(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    boundaries: dict[Node, str],
    stage: int,
    *,
    execution: ChainedExecution | None = None,
    selected: frozenset[Node] | None = None,
) -> list[str]:
    """Emit collective producers after their inputs and before the next dot.

    ``stage == len(plan.dots)`` emits any producers used only by live-outs.
    Allocations are deliberately separate so a loop can retain the workspace.
    ``selected`` restricts this stage's materializations without renumbering
    their complete-plan bindings. The caller owns source-boundary readiness.
    """
    from . import chained_matmul as chain
    from .chained_sparse_reduction import emit_sparse_reduction

    assert plan.region is not None
    execution = execution or ChainedExecution(plan.threads)
    positions = {node: index for index, node in enumerate(plan.region.nodes)}
    start = positions[plan.dots[stage - 1]] if stage else -1
    stop = (
        positions[plan.dots[stage]]
        if stage < len(plan.dots)
        else len(plan.region.nodes)
    )
    eligible = {
        node: name
        for node, name in collective_bindings(plan).items()
        if start < positions[node] < stop
    }
    if selected is not None and not selected <= eligible.keys():
        raise chain._UnsupportedChain(
            "selected collective is not eligible at this stage"
        )
    lines: list[str] = []
    for node, name in eligible.items():
        if selected is not None and node not in selected:
            continue
        operation = classify_collective(node)
        if operation is None:
            raise chain._UnsupportedChain("unsupported region collective")
        if operation.kind == "sum":
            sparse = emit_sparse_reduction(
                cg, plan, boundaries, node, name, execution=execution
            )
            if sparse is not None:
                # The original workspace planner keeps inputs live through this
                # producer and its output disjoint. Only zero-only branches and
                # thread ownership change; publication stays at this graph cut.
                lines.extend([*sparse, execution.sync])
                boundaries[node] = name
                continue
        shape = chain._shape(operation.source)
        extent = shape[operation.axis]
        vectors = shape[1 - operation.axis] if len(shape) == 2 else 1
        vector, position = f"{name}_vector", f"{name}_position"
        coords = _coordinates(operation.axis, vector, position, len(shape))
        expression = chain._Expression(cg, plan, boundaries)
        expression.coordinate_names.update(coords)
        value = expression.value(operation.source, coords)
        domain = chain._operand_domain(cg, operation.source, coords, plan)
        value = chain._masked_operand(value, "cutlass.Float32", domain)
        kind: Literal["sum", "serial_scan", "warp_scan"]
        if (
            operation.kind == "scan"
            and cg.device_function.config.config.get(
                "cute_chained_scan_schedule", "serial"
            )
            == "warp"
        ):
            kind = "warp_scan"
        elif operation.kind == "scan":
            kind = "serial_scan"
        else:
            kind = "sum"
        output_coords = (
            _coordinates(operation.axis, vector, "0", len(shape))
            if node.meta["val"].ndim == 2
            else (vector,)
        )
        phase = emit_producer_phase(
            (
                CollectiveProducer(
                    name,
                    kind,
                    extent,
                    vectors,
                    coords,
                    output_coords,
                    tuple(expression.lines),
                    value,
                    execution,
                ),
            ),
            ast.parse(execution.sync).body[0],
        )
        if phase is None:
            raise chain._UnsupportedChain("unsupported collective producer phase")
        lines.extend(
            ast.unparse(ast.fix_missing_locations(statement)) for statement in phase
        )
        boundaries[node] = name
    return lines
