"""Original prefix arithmetic retained at original operand coordinates.

The caller owns the complete phase/lease proof and native targets. This emitter
does not allocate shared memory, publish a partial prefix boundary, or activate
host vectorization. Every surviving expression uses the ordinary lowering.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import cast

import torch

from ..compile_environment import CompileEnvironment
from . import chained_matmul as chain
from .chained_collectives import emit_collectives_before
from .chained_collectives import warp_prefix_point
from .chained_sparse_reduction import plan_sparse_reduction
from .chained_vector_group import _conjunction
from .chained_vector_native import emit_native_inputs
from .chained_vector_ownership import plan_vector_ownership
from .chained_vector_stage import VectorStageOperand

if TYPE_CHECKING:
    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_broadcast_retention import BroadcastRetentionAttempt
    from .chained_execution import ChainedExecution
    from .chained_matmul import ChainedMatmulPlan
    from .chained_native_read_inputs import NativeReadInput
    from .chained_pointwise_unroll import BoundedProducerUnroll
    from .chained_scan_producer import ScanProducer
    from .chained_vector_native import NativeReadInputs
    from .chained_vector_ownership import VectorOwnership


@dataclass(frozen=True)
class ScanProducerEmission:
    lines: tuple[str, ...]
    shared_outputs: bool
    native_reads: bool = False
    native_sources: tuple[NativeReadInput, ...] = ()
    ownership: VectorOwnership | None = None


def emit_scan_producer(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    candidate: ScanProducer,
    boundaries: dict[Node, str],
    operands: tuple[VectorStageOperand, ...],
    *,
    execution: ChainedExecution,
    producer_unroll: BoundedProducerUnroll,
    native_inputs: NativeReadInputs | None = None,
    broadcast: BroadcastRetentionAttempt | None = None,
) -> ScanProducerEmission | None:
    """Emit one complete native fill and the original delayed reductions.

    All prefix reads by the fill must use its own exact scalar coordinates;
    delayed consumers may read only their proven static residual rows. The
    original source masks, typed boundary-zero guards, final operand domains,
    initial +0, five shuffle-up steps and delayed reduction trees remain.
    """
    try:
        return _emit_scan_producer(
            cg,
            plan,
            candidate,
            boundaries,
            operands,
            execution=execution,
            producer_unroll=producer_unroll,
            native_inputs=native_inputs,
            broadcast=broadcast,
        )
    except chain._UnsupportedChain:
        return None


def _emit_scan_producer(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    candidate: ScanProducer,
    boundaries: dict[Node, str],
    operands: tuple[VectorStageOperand, ...],
    *,
    execution: ChainedExecution,
    producer_unroll: BoundedProducerUnroll,
    native_inputs: NativeReadInputs | None = None,
    broadcast: BroadcastRetentionAttempt | None = None,
) -> ScanProducerEmission | None:
    if execution.threads != 128 or not operands:
        return None
    group = candidate.stage.group
    expected = []
    members = tuple(zip(group.stages, group.geometries, group.offsets, strict=True))
    for role in ("a", "b"):
        for index, geometry, offset in members[:1] if role == "a" else members:
            argument, _ = geometry.operand(role, "row", "column")
            expected.append(
                VectorStageOperand(
                    cast("Node", plan.dots[index].args[argument]),
                    geometry,
                    role,
                    candidate.shape,
                    f"chain_{group.stages[0]}_{role}",
                    offset if role == "b" else 0,
                )
            )
    if operands != tuple(expected):
        return None
    name = candidate.buffer.name
    tag = f"chain_scan_producer_{candidate.first_event}"
    row, base, element = f"{name}_position", f"{tag}_base", f"{tag}_element"
    column = f"({base} + {element})"
    coords = (row, column)
    height, width = candidate.shape
    local = dict(boundaries)
    local[candidate.scan.node] = name
    # The late read proof uses original expression propagation. Merely sharing
    # ancestry or shape cannot authorize a differently indexed prefix read.
    probes, output_coords, predicates = [], [], []
    prefix_read = False
    for operand in operands:
        coordinates = operand.geometry.operand(operand.role, row, column)[1]
        probe = chain._Expression(cg, plan, local)
        probe.coordinate_names.update((row, base, element))
        probe.value(operand.node, coordinates)
        reads = [
            coordinate for node, coordinate in probe.memo if node is candidate.scan.node
        ]
        if any(coordinate != coords for coordinate in reads) or probe.loaded_inputs:
            return None
        prefix_read |= bool(reads)
        probes.append(probe)
        output_coords.append(coordinates)
        predicates.append(
            tuple(
                f"(({coord}) < {extent})"
                for coord, extent in zip(
                    coordinates, chain._shape(operand.node), strict=True
                )
            )
        )
    if not prefix_read or any(
        _conjunction(item) != _conjunction(predicates[0]) for item in predicates[1:]
    ):
        return None
    shared = Counter(
        key
        for probe in probes
        for key in probe.memo
        if key[0] not in probe.boundaries and key[0] not in probe.fragments
    )
    shared_outputs = any(
        count > 1 and chain._pointwise_inputs(key[0]) for key, count in shared.items()
    )
    for action in candidate.deferred:
        sparse = plan_sparse_reduction(plan, action.nodes[0])
        if sparse is None or sparse.position not in candidate.residual_rows:
            return None
        vector = f"{action.writes[0]}_vector"
        probe = chain._Expression(cg, plan, local)
        probe.coordinate_names.add(vector)
        probe.value(sparse.source, (str(sparse.position), vector))
        reads = [
            coordinate for node, coordinate in probe.memo if node is candidate.scan.node
        ]
        if not reads or any(
            coordinate != (str(sparse.position), vector) for coordinate in reads
        ):
            return None
        # Also retain the ordinary source-domain proof for deferred operations.
        chain._operand_domain(cg, sparse.source, (str(sparse.position), vector), plan)
    source = chain._Expression(cg, plan, boundaries)
    source.coordinate_names.update((row, base, element))
    value = source.value(candidate.scan.source, coords)
    native = None
    ownership = None
    if native_inputs is not None:
        ownership = plan_vector_ownership(
            candidate.shape,
            execution.threads,
            tile_columns=32,
            thread_order="column_major",
        )
        # The existing warp-prefix schedule has one complete row tile. The
        # transport may change its storage reads, never its lane ownership.
        if ownership is not None and height == ownership.thread_rows:
            native = emit_native_inputs(
                plan,
                boundaries,
                (source, *probes),
                native_inputs,
                ownership,
                tag,
                row,
                base,
                element,
                execution,
            )
        if native is not None:
            source = chain._Expression(cg, plan, boundaries)
            source.coordinate_names.update((row, base, element))
            source.memo.update(native.replacements)
            value = source.value(candidate.scan.source, coords)
    # Original scalar coefficient loads are part of the scan point program.
    # They retain their own index/mask lowering; only frame reads are moved
    # under the separately validated effective lease schedule.
    value = chain._masked_operand(
        value,
        "cutlass.Float32",
        chain._operand_domain(cg, candidate.scan.source, coords, plan),
    )
    point = warp_prefix_point(name, height, source.lines, value, execution=execution)
    acc = f"{name}_acc"
    retained = None
    if broadcast is not None:
        broadcast_ownership = plan_vector_ownership(
            candidate.shape,
            execution.threads,
            tile_columns=32,
            thread_order="column_major",
        )
        if (
            broadcast_ownership is not None
            and height == broadcast_ownership.thread_rows
        ):
            retained = broadcast.emit(
                cg,
                plan,
                boundaries,
                probes,
                broadcast_ownership,
                tag=tag,
                row=row,
                base=base,
                element=element,
                execution=execution,
            )
    output = chain._Expression(cg, plan, local)
    output.coordinate_names.update((row, base, element))
    if native is not None:
        output.memo.update(native.replacements)
    if retained is not None:
        output.memo.update(retained.replacements)
    output.fragments[candidate.scan.node] = (
        coords,
        chain._materialized_value(
            name, candidate.shape, coords, "cutlass.Float32", storage_value=acc
        ),
    )
    setup, stores, zeros, copies = [], [], [], []
    for index, (operand, coordinates) in enumerate(
        zip(operands, output_coords, strict=True)
    ):
        if operand.node.meta["val"].dtype not in (torch.bfloat16, torch.float16):
            return None
        dtype = CompileEnvironment.current().backend.dtype_str(
            operand.node.meta["val"].dtype
        )
        result = output.value(operand.node, coordinates)
        result = chain._masked_operand(
            result, dtype, chain._operand_domain(cg, operand.node, coordinates, plan)
        )
        prefix = f"{tag}_output_{index}"
        destination = (
            operand.target
            if operand.offset == 0
            else f"cute.domain_offset(({operand.offset}, 0), {operand.target})"
        )
        setup.extend(
            [
                f"{prefix}_copy = cute.make_tiled_copy_tv(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), {dtype}, num_bits_per_copy=128), cute.make_layout((32, 4), stride=(1, 32)), cute.make_layout((1, 8)))",
                f"{prefix}_thread = {prefix}_copy.get_slice({execution.thread})",
                f"{prefix}_target = {prefix}_thread.partition_D({destination})",
                f"{prefix}_values = cute.make_rmem_tensor({prefix}_target[None, 0, 0].shape, {dtype})",
            ]
        )
        stores.append(f"{prefix}_values[{element}] = {result}")
        zeros.append(f"{prefix}_values[{element}] = {dtype}(0)")
        copies.append(
            f"cute.copy({prefix}_copy, {prefix}_values, {prefix}_target[None, 0, {tag}_step])"
        )
    residual = " or ".join(
        f"{row} == {position}" for position in candidate.residual_rows
    )
    body = [
        f"{acc} = cutlass.Float32(0)",
        f"for {name}_part in cutlass.range_constexpr(1):",
        chain._indent(point),
        f"if {residual}:",
        f"    {name}[{row}, {column}] = {acc}",
        f"if {' & '.join(predicates[0])}:",
        chain._indent([*output.lines, *stores]),
        "else:",
        chain._indent(zeros),
    ]
    tail = []
    for action in candidate.deferred:
        assert action.source_stage is not None
        tail.extend(
            emit_collectives_before(
                cg,
                plan,
                local,
                action.source_stage,
                execution=execution,
                selected=frozenset(action.nodes),
            )
        )
    trips = width // 32
    lines = [
        *(native.setup if native is not None else ()),
        *setup,
        *(retained.before_steps if retained is not None else ()),
        f"for {tag}_step in cutlass.range({trips}, unroll={producer_unroll.loop_factor(trips)}):",
        f"    {row} = {execution.thread} % 32",
        f"    {base} = ({execution.thread} // 32 + {tag}_step * 4) * 8",
        *("    " + line for line in (native.loads if native is not None else ())),
        *(
            [chain._indent(retained.per_step, 4)]
            if retained is not None and retained.per_step
            else []
        ),
        f"    for {element} in cutlass.range_constexpr(8):",
        chain._indent(body, 8),
        chain._indent(copies, 4),
        execution.sync,
        *tail,
    ]
    if retained is not None and broadcast is not None:
        broadcast.complete(retained, boundaries, execution, lines)
    return ScanProducerEmission(
        tuple(lines),
        shared_outputs,
        native is not None,
        () if native is None else native.sources,
        None if native is None else ownership,
    )
