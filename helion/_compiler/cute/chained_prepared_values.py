"""Typed shared images crossing a graph-derived preparation/recurrence cut.

The frame planner owns byte lifetimes. These helpers only construct views and
materialize the original expressions; the caller owns slot readiness/release.
Scalar images occupy one element, and predicates occupy one byte per element.
No implicit packed-i1 pointer is allowed to change the frame's byte accounting.
"""

from __future__ import annotations

from dataclasses import replace
import math
from typing import TYPE_CHECKING

import torch

from ..compile_environment import CompileEnvironment
from .chained_matmul import _Expression
from .chained_matmul import _indent
from .chained_matmul import _shape
from .chained_scratch_layout import xor_swizzle
from .chained_vector_expression import emit_vector_expression
from .chained_vector_leaf import DenseVectorSink

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_broadcast_retention import BroadcastOptions
    from .chained_cache_layout import PointwiseCacheLayouts
    from .chained_execution import ChainedExecution
    from .chained_matmul import ChainedMatmulPlan
    from .chained_pointwise_unroll import BoundedProducerUnroll
    from .chained_preparation_frame import PreparationBuffer
    from .chained_preparation_frame import PreparationFrame
    from .chained_preparation_leaves import PreparationLeaf
    from .chained_preparation_reads import PreparationFrontierCompletion
    from .chained_preparation_storage import BoundPreparationStorage
    from .chained_prepared_groups import PreparedGroupBinding
    from .chained_prepared_operands import PreparedOperand
    from .chained_scan_producer import ScanProducer
    from .chained_scratch_layout import ScratchLayouts
    from .chained_vector_native import NativeReadInputs
    from .chained_vector_stage import VectorStaging


def storage_dtype(dtype: torch.dtype) -> str:
    return (
        "cutlass.Uint8"
        if dtype == torch.bool
        else CompileEnvironment.current().backend.dtype_str(dtype)
    )


def buffer_layout(
    buffer: PreparationBuffer,
    scratch: ScratchLayouts,
    cache_layouts: PointwiseCacheLayouts | None = None,
) -> str:
    shape = buffer.shape or (1,)
    if buffer.dtype == torch.float32:
        layout = scratch.layout(buffer.name, shape)
    else:
        strides = tuple(math.prod(shape[index + 1 :]) for index in range(len(shape)))
        layout = f"cute.make_layout({shape!r}, stride={strides!r})"
    if buffer.kind == "cache" and cache_layouts is not None:
        return cache_layouts.layout(shape, buffer.dtype, layout)
    return layout


def bind_frame_buffers(
    frame: PreparationFrame,
    byte_pointer: str,
    scratch: ScratchLayouts,
    *,
    prepared_operands: tuple[PreparedOperand, ...] = (),
    prepared_groups: tuple[PreparedGroupBinding, ...] = (),
    prepared_leaves: tuple[PreparationLeaf, ...] = (),
    cache_layouts: PointwiseCacheLayouts | None = None,
    scan_producer: ScanProducer | None = None,
    preparation_storage: BoundPreparationStorage | None = None,
) -> list[str]:
    """Bind views, without reading, writing, allocating, or publishing a slot.

    A/B views belong to the MMA stage emitter. All other names match the
    complete-plan collective/cache/C symbols, irrespective of source ordering.
    ``byte_pointer`` must point to the selected exclusively owned frame base.
    """
    if preparation_storage is not None:
        pipeline = preparation_storage.physical.accepted.pipeline
        if (
            pipeline.frame is not frame
            or prepared_operands != pipeline.prepared_operands
            or prepared_groups != pipeline.prepared_groups
            or prepared_leaves != pipeline.prepared_leaves
            or scan_producer is not pipeline.scan_producer
            or cache_layouts != preparation_storage.cache_layouts
        ):
            raise ValueError("physical binding disagrees with the semantic frame")
        return preparation_storage.view_lines(byte_pointer, scratch)
    lines = []
    # Native/grouped/leaf views never inherit an earlier dense-view admission.
    for buffer in frame.buffers:
        scratch.vector_sinks.pop(buffer.name, None)
    offsets = {}
    if scan_producer is not None:
        if (
            scan_producer.revision.frame is not frame
            or scan_producer._selection != scan_producer._fields()
            or scan_producer.scratch_mode != scratch.mode
        ):
            raise ValueError("scan producer view proof changed")
        offsets = dict(scan_producer.overrides)
    native = {operand.buffer.name: operand for operand in prepared_operands}
    grouped = {}
    for binding in prepared_groups:
        candidate = binding.candidate
        dtype = storage_dtype(candidate.dtype)
        for member in candidate.members:
            region = frame.layout.region(member.buffer.name)
            if (
                member.buffer.name in grouped
                or member.buffer.name in native
                or member.buffer not in frame.buffers
                or region.byte_offset != binding.byte_offset + member.byte_offset
                or region.byte_size
                != math.prod(member.physical_shape) * candidate.dtype.itemsize
            ):
                raise ValueError("prepared group does not match the frame allocation")
            grouped[member.buffer.name] = member
        lines.extend(
            [
                f"{candidate.name}_ptr = cute.recast_ptr({byte_pointer} + {binding.byte_offset}, dtype={dtype})",
                *candidate.native_layout,
            ]
        )
    for buffer in frame.buffers:
        if buffer.kind in ("a", "b"):
            continue
        region = frame.layout.region(buffer.name)
        dtype = storage_dtype(buffer.dtype)
        pointer = f"cute.recast_ptr({byte_pointer} + {offsets.get(buffer.name, region.byte_offset)}, dtype={dtype})"
        if buffer.kind == "leaf":
            leaf = next(
                (leaf for leaf in prepared_leaves if leaf.node is buffer.node), None
            )
            if leaf is None or leaf.name != buffer.name:
                raise ValueError("prepared leaf does not match the frame allocation")
            lines.extend(leaf.view(byte_pointer, region.byte_offset))
            continue
        if buffer.name in native:
            operand = native[buffer.name]
            if operand.buffer != buffer or operand.region != region:
                raise ValueError("prepared operand does not match the frame allocation")
            lines.extend([f"{buffer.name}_ptr = {pointer}", *operand.native_layout])
            continue
        if buffer.name in grouped:
            lines.extend(
                [f"{buffer.name}_ptr = {pointer}", *grouped[buffer.name].native_layout]
            )
            continue
        layout = buffer_layout(buffer, scratch, cache_layouts)
        lines.append(f"{buffer.name} = cute.make_tensor({pointer}, {layout})")
        if (
            buffer.kind == "frontier"
            and len(buffer.shape) == 2
            and buffer.dtype in (torch.bfloat16, torch.float16, torch.float32)
            and not (
                buffer.dtype == torch.float32
                and scratch.mode == "xor"
                and xor_swizzle(buffer.shape) is not None
            )
        ):
            # This is the actual plain-view branch above, not vector_store's
            # weaker native-copy admission. FP32 XOR is explicitly excluded;
            # native/grouped/leaf bindings have already continued.
            scratch.vector_sinks[buffer.name] = DenseVectorSink(
                buffer.name, (buffer.shape[0], buffer.shape[1]), dtype
            )
    return lines


def frontier_bindings(frame: PreparationFrame) -> dict[Node, str]:
    return {
        buffer.node: buffer.name
        for buffer in frame.buffers
        if buffer.kind == "frontier" and buffer.node is not None
    }


def emit_prepared_value(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    buffer: PreparationBuffer,
    boundaries: dict[Node, str],
    execution: ChainedExecution,
    *,
    vector: VectorStaging | None = None,
    producer_unroll: BoundedProducerUnroll | None = None,
    vector_store: bool = True,
    raw_boundaries: Mapping[Node, str] | None = None,
    native_inputs: NativeReadInputs | None = None,
    completion: PreparationFrontierCompletion | None = None,
    shared_sink: DenseVectorSink | None = None,
) -> list[str]:
    """Publish an exact typed expression, including an existing boundary read.

    Do not redirect preparation-side uses to this new image: the frame planner
    computed their lifetimes against the original C/collective/cache boundaries.
    Only the ordered role uses the complete ``frontier_bindings`` mapping.
    """
    node = buffer.node
    inputs = tuple(boundaries.items()) if completion is not None else ()
    assert node is not None and buffer.kind == "frontier"
    assert _shape(node) == buffer.shape
    if vector is not None and vector.enabled and len(buffer.shape) == 2:
        # A dynamic tiled-copy partition of an XOR view needs a power-of-two
        # thread tile. The preparation role may contain e.g. twelve warps;
        # let its largest power-of-two prefix own the copy, while every role
        # participant still reaches the original publication barrier below.
        copy_threads = (
            1 << (execution.threads.bit_length() - 1)
            if vector_store
            else execution.threads
        )
        options: BroadcastOptions = {}
        if vector.broadcast is not None:
            options["broadcast"] = vector.broadcast
        broadcast_first = (
            0 if vector.broadcast is None else vector.broadcast.checkpoint()
        )
        lines = emit_vector_expression(
            cg,
            plan,
            boundaries,
            node,
            shape=(buffer.shape[0], buffer.shape[1]),
            coordinates=lambda row, column: (row, column),
            offset=0,
            tag=f"{buffer.name}_vector",
            target=buffer.name,
            vector_store=vector_store,
            producer_unroll=producer_unroll,
            execution=replace(execution, threads=copy_threads),
            raw_boundaries=raw_boundaries,
            native_inputs=native_inputs,
            shared_sink=shared_sink if vector.async_enabled else None,
            async_staging=vector,
            **options,
        )
        if lines is not None:
            vector.activated = True
            broadcast_placement = (
                None
                if vector.broadcast is None
                else vector.broadcast.place(
                    broadcast_first,
                    lines,
                    1 if copy_threads != execution.threads else 0,
                    indent=4 if copy_threads != execution.threads else 0,
                )
            )
            if copy_threads != execution.threads:
                lines = [
                    f"if {execution.thread} < {copy_threads}:",
                    _indent(lines, 4),
                ]
            lines = [*lines, execution.sync]
            if completion is not None:
                completion.record(
                    cg, plan, (buffer,), execution, inputs, boundaries, lines
                )
            if vector.broadcast is not None:
                vector.broadcast.enclose(broadcast_first, lines, (broadcast_placement,))
            return lines
    size = math.prod(buffer.shape)
    prefix = buffer.name
    offset = f"{prefix}_offset"
    coords = tuple(
        f"({offset} // {math.prod(buffer.shape[index + 1 :])}) % {extent}"
        for index, extent in enumerate(buffer.shape)
    )
    expression = _Expression(cg, plan, boundaries)
    expression.coordinate_names.add(offset)
    value = expression.value(node, coords)
    dtype = storage_dtype(buffer.dtype)
    index = ", ".join(coords) if coords else "0"
    lines = [
        f"for {prefix}_step in cutlass.range_constexpr({(size + execution.threads - 1) // execution.threads}):",
        f"    {offset} = {execution.thread} + {prefix}_step * {execution.threads}",
        f"    if {offset} < {size}:",
        _indent(expression.lines, 8),
        f"        {prefix}[{index}] = {dtype}({value})",
        execution.sync,
    ]
    if completion is not None:
        completion.record(cg, plan, (buffer,), execution, inputs, boundaries, lines)
    return lines
