"""Retain original coordinate-owned values through row sums and operand fills.

The frame and its native operand views remain caller-owned. This emitter only
changes traversal and register lifetime: a warp reduces one row in the original
lane/part order, then publishes that row's operands with ordinary scalar stores.
It neither allocates shared storage nor selects a vector staging optimization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import torch

from ..compile_environment import CompileEnvironment
from . import chained_matmul as chain
from .chained_row_collectives import plan_row_collective_group
from .chained_sparse_reduction import emit_sparse_reduction
from .chained_vector_group import _conjunction
from .chained_vector_stage import VectorStageOperand

if TYPE_CHECKING:
    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_execution import ChainedExecution
    from .chained_matmul import ChainedMatmulPlan
    from .chained_pointwise_unroll import BoundedProducerUnroll
    from .chained_pointwise_unroll import PointwiseUnroll
    from .chained_preparation_frame import PreparationFrame
    from .chained_row_collectives import RowCollectiveGroup


def _operands_match(
    plan: ChainedMatmulPlan,
    candidate: RowCollectiveGroup,
    operands: tuple[VectorStageOperand, ...],
) -> bool:
    """Require the complete original A image and ordered, disjoint B members."""
    group = candidate.stage.group
    prefix = f"chain_{group.stages[0]}"
    expected = []
    if any(type(operand.offset) is not int for operand in operands):
        return False
    members = tuple(zip(group.stages, group.geometries, group.offsets, strict=True))
    for role in ("a", "b"):
        for index, geometry, offset in members[:1] if role == "a" else members:
            arg, _ = geometry.operand(role, "row", "column")
            expected.append(
                VectorStageOperand(
                    cast("Node", plan.dots[index].args[arg]),
                    geometry,
                    role,
                    candidate.shape,
                    f"{prefix}_{role}",
                    offset if role == "b" else 0,
                )
            )
    return operands == tuple(expected)


def emit_row_collective_group(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    frame: PreparationFrame,
    candidate: RowCollectiveGroup,
    boundaries: dict[Node, str],
    operands: tuple[VectorStageOperand, ...],
    *,
    execution: ChainedExecution,
    producer_unroll: BoundedProducerUnroll | PointwiseUnroll | None,
) -> list[str] | None:
    """Attempt retention without installing partial boundaries or statements.

    The caller must invoke this after final frame/native view binding, just
    before the candidate's first action. On success it skips only the admitted
    sums and fill and installs their sum boundaries atomically. It retains the
    stage's existing shared-view fence, whole-role barrier and complete MMA.
    The returned tail contains one original barrier per collective, outside all
    ownership predicates. No vector-staging activation is performed.

    Probes can consume compiler temporary names, like ordinary expression
    probes. Failed admission returns None before unroll activation or boundary
    mutation. Arithmetic uses the existing compiler/NVVM contraction policy;
    sharing source values is not a promise of cross-config bitwise equality.
    """
    try:
        return _emit_row_collective_group(
            cg,
            plan,
            frame,
            candidate,
            boundaries,
            operands,
            execution=execution,
            producer_unroll=producer_unroll,
        )
    except chain._UnsupportedChain:
        return None


def _emit_row_collective_group(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    frame: PreparationFrame,
    candidate: RowCollectiveGroup,
    boundaries: dict[Node, str],
    operands: tuple[VectorStageOperand, ...],
    *,
    execution: ChainedExecution,
    producer_unroll: BoundedProducerUnroll | PointwiseUnroll | None,
) -> list[str] | None:
    if plan.region is None:
        return None
    shapes = {
        node: chain._shape(node)
        for node in plan.region.nodes
        if isinstance(node.meta.get("val"), torch.Tensor)
    }
    current = plan_row_collective_group(plan, frame, candidate.first_event, shapes)
    if current != candidate or not _operands_match(plan, candidate, operands):
        return None
    # Current warp-stage geometry requires at least one complete warpgroup.
    if execution.threads % 128:
        return None
    by_name = {buffer.name: buffer for buffer in frame.buffers}
    for item in candidate.read_regions:
        buffer = by_name[item.original.name]
        if buffer.node is None or boundaries.get(buffer.node) != buffer.name:
            return None
    local_boundaries = dict(boundaries)
    for operation, buffer in zip(candidate.collectives, candidate.buffers, strict=True):
        keepdim = (
            operation.node.args[2]
            if len(operation.node.args) == 3
            else operation.node.kwargs.get("keepdim", False)
        )
        if (
            type(keepdim) is not bool
            or operation.node.kwargs.get("dtype") not in (None, torch.float32)
            or shapes[operation.node]
            != ((candidate.shape[0], 1) if keepdim else (candidate.shape[0],))
            or emit_sparse_reduction(
                cg,
                plan,
                local_boundaries,
                operation.node,
                buffer.name,
                execution=execution,
            )
            is not None
        ):
            return None
        local_boundaries[operation.node] = buffer.name
    tag = f"chain_row_collective_{candidate.first_event}"
    row, column, part = f"{tag}_row", f"{tag}_column", f"{tag}_part"
    coords = (row, column)
    height, width = candidate.shape
    parts = (width + 31) // 32
    sum_fragments = {}
    for index, operation in enumerate(candidate.collectives):
        output_coords = (row, "0") if len(shapes[operation.node]) == 2 else (row,)
        # Preserve the exact typed shared-boundary validity guard, substituting
        # only its scalar read with the original lane-zero broadcast.
        bounds = " and ".join(
            f"0 <= ({coord}) < {extent}"
            for coord, extent in zip(output_coords, shapes[operation.node], strict=True)
        )
        sum_fragments[operation.node] = (
            output_coords,
            f"({tag}_sum_{index} if {bounds} else cutlass.Float32(0))",
        )

    def expression(*, with_sums: bool = False) -> chain._Expression:
        result = chain._Expression(cg, plan, local_boundaries)
        result.coordinate_names.update(coords)
        if with_sums:
            result.fragments.update(sum_fragments)
        return result

    reduction_probes = []
    for operation in candidate.collectives:
        probe = expression()
        probe.value(operation.source, coords)
        reduction_probes.append(probe)
    output_probes = []
    output_coords = []
    predicates = []
    for operand in operands:
        coordinates = operand.geometry.operand(operand.role, row, column)[1]
        probe = expression(with_sums=True)
        probe.value(operand.node, coordinates)
        output_probes.append(probe)
        output_coords.append(coordinates)
        predicates.append(
            [
                f"(({coord}) < {extent})"
                for coord, extent in zip(coordinates, shapes[operand.node], strict=True)
            ]
        )
    if any(
        _conjunction(predicate) != _conjunction(predicates[0])
        for predicate in predicates[1:]
    ):
        return None
    reduction_keys = set().union(*(probe.memo.keys() for probe in reduction_probes))
    output_keys = set().union(*(probe.memo.keys() for probe in output_probes))
    keys = reduction_keys & output_keys
    choices = {
        node
        for node, coordinate in keys
        if coordinate == coords
        and shapes[node] == candidate.shape
        and node.meta["val"].dtype in (torch.bfloat16, torch.float16, torch.float32)
        and node not in sum_fragments
    }
    all_keys = reduction_keys | output_keys
    choices = {
        node
        for node in choices
        if all(coordinate == coords for other, coordinate in all_keys if other is node)
    }
    # Keep maximal shared original values, not both an expression and all of
    # its input snapshots. No formula matching or arithmetic CSE is performed.
    roots = tuple(
        node
        for node in plan.region.nodes
        if node in choices
        and not any(
            node in chain._ancestors(other) for other in choices if other is not node
        )
    )
    if not roots:
        return None
    for node in roots:
        # Prove the original domain is expressible, but do not mask a cached
        # intermediate: nonlinear operations on a masked load's zero must
        # still execute. Original domains are applied only at the same sum and
        # final operand boundaries as ordinary lowering.
        chain._operand_domain(cg, node, coords, plan)
    first_use = {
        node: next(
            index
            for index, probe in enumerate(reduction_probes)
            if (node, coords) in probe.memo
        )
        for node in roots
    }
    names = {node: f"{tag}_retained_{index}" for index, node in enumerate(roots)}
    dtype_str = CompileEnvironment.current().backend.dtype_str
    setup = [
        f"{names[node]} = cute.make_rmem_tensor(({parts},), {dtype_str(node.meta['val'].dtype)})"
        for node in roots
    ]
    body = []
    fragments = {}
    for index, (operation, buffer) in enumerate(
        zip(candidate.collectives, candidate.buffers, strict=True)
    ):
        evaluate = expression()
        evaluate.fragments.update(fragments)
        retained = []
        for node in roots:
            if first_use[node] != index:
                continue
            value = evaluate.value(node, coords)
            retained.append(f"{names[node]}[{part}] = {value}")
            fragments[node] = (coords, f"{names[node]}[{part}]")
        source = expression()
        source.fragments.update(fragments)
        value = source.value(operation.source, coords)
        value = chain._masked_operand(
            value,
            "cutlass.Float32",
            chain._operand_domain(cg, operation.source, coords, plan),
        )
        accumulator = f"{tag}_acc_{index}"
        body.extend(
            [
                f"{accumulator} = cutlass.Float32(0)",
                f"for {part} in cutlass.range_constexpr({parts}):",
                f"    {column} = {execution.thread} % 32 + {part} * 32",
                f"    if {column} < {width}:",
                chain._indent(
                    [
                        *evaluate.lines,
                        *retained,
                        *source.lines,
                        f"{accumulator} += {value}",
                    ],
                    8,
                ),
                *(
                    f"{accumulator} += cute.arch.shuffle_sync_down({accumulator}, offset={offset})"
                    for offset in (16, 8, 4, 2, 1)
                ),
                f"if {execution.thread} % 32 == 0:",
                f"    {buffer.name}[{', '.join(sum_fragments[operation.node][0])}] = {accumulator}",
                f"{tag}_sum_{index} = cute.arch.shuffle_sync({accumulator}, 0)",
            ]
        )
    output = expression(with_sums=True)
    output.fragments.update(fragments)
    stores = []
    zeros = []
    for operand, coordinates in zip(operands, output_coords, strict=True):
        value = output.value(operand.node, coordinates)
        dtype = dtype_str(operand.node.meta["val"].dtype)
        # This is exactly the ordinary VectorStageOperand final-value callback.
        value = chain._masked_operand(
            value, dtype, chain._operand_domain(cg, operand.node, coordinates, plan)
        )
        target = f"{operand.target}[{row} + {operand.offset}, {column}]"
        stores.append(f"{target} = {value}")
        zeros.append(f"{target} = {dtype}(0)")
    body.extend(
        [
            f"for {part} in cutlass.range_constexpr({parts}):",
            f"    {column} = {execution.thread} % 32 + {part} * 32",
            f"    if {column} < {width}:",
            f"        if {' & '.join(predicates[0])}:",
            chain._indent([*output.lines, *stores], 12),
            "        else:",
            chain._indent(zeros, 12),
        ]
    )
    warps = execution.threads // 32
    trips = (height + warps - 1) // warps
    unroll = 1 if producer_unroll is None else producer_unroll.loop_factor(trips)
    return [
        *setup,
        f"for {tag}_step in cutlass.range({trips}, unroll={unroll}):",
        f"    {row} = {execution.thread} // 32 + {tag}_step * {warps}",
        f"    if {row} < {height}:",
        chain._indent(body, 8),
        *[execution.sync for _ in candidate.collectives],
    ]
