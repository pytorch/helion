"""Loop ports and resident state for the common contraction-region emitter.

The body is an ordinary contraction DAG. This module binds its lexical inputs,
keeps carried tiles at their declared precision, and exports the final values.
It does not recognize coefficient formulas or replace any arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import math
import operator
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.fx import Node

from ...language import _tracing_ops
from ...language import memory_ops
from ...language.matmul_ops import dot
from ..compile_environment import CompileEnvironment
from ..compile_environment import RuntimeInputSpecialization
from ..device_ir import ForLoopGraphInfo
from ..device_ir import HelperFunctionGraphInfo
from ..device_ir import RootGraphInfo
from .chained_execution import ChainedExecution
from .contraction_region import collect_contraction_region

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    from torch._dynamo.source import Source

    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST
    from .chained_matmul import ChainedMatmulPlan
    from .chained_scratch_layout import ScratchLayouts
    from .contraction_region import ContractionRegion
    from .warp_specialized_plan import SharedMemoryLayoutPlan


@dataclass(frozen=True)
class ChainedLoopPlan:
    root: RootGraphInfo
    body: ForLoopGraphInfo
    region: ContractionRegion
    call: Node
    inputs: tuple[Node, ...]
    begin: object
    end: object
    final_values: tuple[tuple[Node, int], ...]
    final_stores: tuple[Node, ...]

    @property
    def block_id(self) -> int:
        return self.body.block_ids[0]

    @property
    def storage_key(self) -> str:
        return f"cute_contraction_loop_storage_v1_{self.root.graph_id}_{self.body.graph_id}"

    @staticmethod
    def carry_name(index: int) -> str:
        return f"chain_loop_carry_{index}"

    def boundaries(self) -> dict[Node, str]:
        result = {
            carry.input: self.carry_name(carry.input_index)
            for carry in self.region.carries
        }
        result.update(
            (node, self.carry_name(index)) for node, index in self.final_values
        )
        return result

    def captures(self) -> dict[Node, str]:
        carried = {carry.input_index for carry in self.region.carries}
        return {
            node: f"chain_loop_capture_{index}"
            for index, node in enumerate(self.body.graph.find_nodes(op="placeholder"))
            if index not in carried
        }


def discover_chained_loop(graphs: Sequence[GraphInfo]) -> ChainedLoopPlan | None:
    """Recognize a lexical contraction loop with scalar captures and tile carries."""
    roots = [info for info in graphs if isinstance(info, RootGraphInfo)]
    bodies = [info for info in graphs if isinstance(info, ForLoopGraphInfo)]
    if (
        len(roots) != 1
        or len(bodies) != 1
        or any(
            not isinstance(
                info, (RootGraphInfo, ForLoopGraphInfo, HelperFunctionGraphInfo)
            )
            for info in graphs
        )
    ):
        return None
    root, body = roots[0], bodies[0]
    region = collect_contraction_region(body)
    if (
        region is None
        or region.loop_interface is None
        or not region.carries
        or len(body.block_ids) != 1
        or any(node.target is dot for node in root.graph.nodes)
    ):
        return None
    helpers = {
        info.graph_id for info in graphs if isinstance(info, HelperFunctionGraphInfo)
    }
    # Helpers are owned by the body's scan operations, not independent device
    # regions. The main contraction classifier proves each combiner's semantics.
    if any(
        len(scan.args) != 5
        or type(scan.args[0]) is not int
        or scan.args[0] not in helpers
        for scan in region.scans
    ) or helpers != {scan.args[0] for scan in region.scans}:
        return None
    calls = [node for node in root.graph.nodes if node.target is _tracing_ops._for_loop]
    if len(calls) != 1:
        return None
    call = calls[0]
    if (
        len(call.args) != 4
        or call.args[0] != body.graph_id
        or not isinstance(call.args[1], (list, tuple))
        or not isinstance(call.args[2], (list, tuple))
        or len(call.args[1]) != 1
        or len(call.args[2]) != 1
        or not isinstance(call.args[3], (list, tuple))
        or len(call.args[3]) != region.loop_interface.input_count
        or not all(isinstance(node, Node) for node in call.args[3])
    ):
        return None
    inputs = cast("tuple[Node, ...]", tuple(call.args[3]))
    carried = {carry.input_index for carry in region.carries}
    placeholders = tuple(body.graph.find_nodes(op="placeholder"))
    for index, (incoming, local) in enumerate(zip(inputs, placeholders, strict=True)):
        value, local_value = incoming.meta.get("val"), local.meta.get("val")
        if (
            not isinstance(value, torch.Tensor)
            or not isinstance(local_value, torch.Tensor)
            or value.dtype != local_value.dtype
            or value.ndim != (2 if index in carried else 0)
        ):
            return None
    final_values: list[tuple[Node, int]] = []
    outputs = {carry.output_index: carry.input_index for carry in region.carries}
    for user in call.users:
        if (
            user.target is not operator.getitem
            or len(user.args) != 2
            or user.args[0] is not call
            or type(user.args[1]) is not int
            or user.args[1] not in outputs
        ):
            return None
        index = outputs[user.args[1]]
        final_values.append((user, index))
        for phi in user.users:
            if phi.target is _tracing_ops._phi:
                if phi.args != (inputs[index], user):
                    return None
                final_values.append((phi, index))
    nodes = tuple(root.graph.nodes)
    position = nodes.index(call)
    if any(node.target is memory_ops.store for node in nodes[:position]):
        return None
    final_stores = tuple(
        node for node in nodes[position + 1 :] if node.target is memory_ops.store
    )
    if not final_stores or not region.stores:
        return None
    return ChainedLoopPlan(
        root,
        body,
        region,
        call,
        inputs,
        call.args[1][0],
        call.args[2][0],
        tuple(final_values),
        final_stores,
    )


def _storage_sources(
    loop: ChainedLoopPlan,
) -> tuple[tuple[Source, ...], tuple[int, ...]] | None:
    """Map external storage to replayable arguments, excluding fresh outputs."""
    env = CompileEnvironment.current()
    accesses = [
        node
        for graph in (loop.root.graph, loop.body.graph)
        for node in graph.nodes
        if node.target in (memory_ops.load, memory_ops.store)
    ]
    sources: list[Source] = []
    writes: set[int] = set()
    reads: set[int] = set()
    fresh_writes: set[int] = set()
    fresh_reads: set[int] = set()
    for node in accesses:
        source = node.args[0]
        if not isinstance(source, Node):
            return None
        if source.target is not _tracing_ops._host_tensor:
            # Internal indexing is an expression read, not another external
            # storage access. Its host-load ancestors are independently present
            # in these graphs and must still participate in the alias proof.
            if node.target is memory_ops.load:
                continue
            return None
        tensor = source.meta["val"]
        storage = tensor.untyped_storage()
        if storage in env.fresh_allocation_storages:
            (fresh_writes if node.target is memory_ops.store else fresh_reads).add(
                storage._cdata
            )
            continue
        candidates = [
            argument
            for value, argument in env.input_sources.items()
            if value.untyped_storage()._cdata == storage._cdata
        ]
        if not candidates or node.target is memory_ops.store and len(candidates) != 1:
            return None
        # Read-only packed views can share storage. Guard every visible view,
        # rather than guessing which host argument a reshaped view came from.
        for argument in candidates:
            if argument not in sources:
                sources.append(argument)
            index = sources.index(argument)
            (writes if node.target is memory_ops.store else reads).add(index)
    if writes & reads or fresh_reads & fresh_writes:
        return None
    return tuple(sources), tuple(sorted(writes))


def _disjoint_writes(values: Sequence[object], *, writes: tuple[int, ...]) -> bool:
    from torch._subclasses.fake_tensor import FakeTensor

    if any(not isinstance(value, torch.Tensor) for value in values):
        return False
    tensors = cast("tuple[torch.Tensor, ...]", tuple(values))
    # External destinations must have a non-overlapping physical layout. Inputs
    # may be strided; conservative storage spans still protect all their reads.
    if any(not tensors[index].is_contiguous() for index in writes):
        return False
    spans = []
    for tensor in tensors:
        if any(stride < 0 for stride in tensor.stride()):
            return False
        size = (
            0
            if tensor.numel() == 0
            else 1
            + sum(
                (extent - 1) * stride
                for extent, stride in zip(tensor.shape, tensor.stride(), strict=True)
            )
        )
        if isinstance(tensor, FakeTensor):
            base = tensor.untyped_storage()._cdata
            start = tensor.storage_offset() * tensor.element_size()
        else:
            base = 0
            start = tensor.data_ptr()
        spans.append((base, start, start + size * tensor.element_size()))
    return all(
        i == j or left[0] != right[0] or left[2] <= right[1] or right[2] <= left[1]
        for i in writes
        for j, right in enumerate(spans)
        for left in (spans[i],)
    )


def register_loop_storage(loop: ChainedLoopPlan) -> bool:
    result = _storage_sources(loop)
    if result is None:
        return False
    sources, writes = result
    CompileEnvironment.current().register_runtime_input_specialization(
        loop.storage_key,
        RuntimeInputSpecialization(
            sources=sources,
            classifier_identity=(loop.storage_key, tuple(map(repr, sources)), writes),
            classifier=partial(_disjoint_writes, writes=writes),
            reusable_tensor_properties=frozenset(("data_ptr", "storage_span")),
        ),
    )
    return True


def loop_storage_is_proven(loop: ChainedLoopPlan) -> bool:
    return CompileEnvironment.current().runtime_input_specialization_matches_bound(
        loop.storage_key, True
    )


def loop_storage_matches_runtime(loop: ChainedLoopPlan) -> bool:
    """Classify this bind before its immutable cache-key facts are published."""
    from ..compile_environment import _replay_tensor_input_source

    env = CompileEnvironment.current()
    specialization = env.runtime_input_specializations[loop.storage_key]
    values = tuple(
        _replay_tensor_input_source(source, env.runtime_arg_values_by_name)
        for source in specialization.sources
    )
    return specialization.classifier(values) is True


def carry_shared_bytes(loop: ChainedLoopPlan) -> int:
    from .chained_matmul import _shape

    return sum(
        (math.prod(_shape(carry.input)) * carry.input.meta["val"].element_size() + 127)
        // 128
        * 128
        for carry in loop.region.carries
    )


def initialize_loop(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    scratch: ScratchLayouts,
    workspace: SharedMemoryLayoutPlan | None = None,
    *,
    carry_pointers: Mapping[str, str] | None = None,
) -> list[str]:
    """Load immutable captures and initial state once, including zero-trip loops."""
    from .chained_matmul import _Expression
    from .chained_matmul import _indent
    from .chained_matmul import _shape

    loop = plan.loop
    assert loop is not None
    env = CompileEnvironment.current()
    lines: list[str] = []
    carried = {carry.input_index for carry in loop.region.carries}
    for index, incoming in enumerate(loop.inputs):
        if index in carried:
            continue
        expression = _Expression(cg, plan, {})
        value = expression.value(incoming, ())
        lines.extend([*expression.lines, f"chain_loop_capture_{index} = {value}"])
    expression = _Expression(cg, plan, {})
    begin, end = expression.scalar(loop.begin), expression.scalar(loop.end)
    lines.extend(
        [*expression.lines, f"chain_loop_begin = {begin}", f"chain_loop_end = {end}"]
    )
    for carry in loop.region.carries:
        shape = _shape(carry.input)
        size = math.prod(shape)
        prefix = loop.carry_name(carry.input_index)
        dtype = env.backend.dtype_str(carry.input.meta["val"].dtype)
        coords = (f"{prefix}_offset // {shape[1]}", f"{prefix}_offset % {shape[1]}")
        expression = _Expression(cg, plan, {})
        expression.coordinate_names.add(f"{prefix}_offset")
        value = expression.value(loop.inputs[carry.input_index], coords)
        layout = scratch.layout(prefix, shape, carry.input.meta["val"].dtype)
        pointer = (
            carry_pointers[prefix]
            if carry_pointers is not None and prefix in carry_pointers
            else f"cute.arch.alloc_smem({dtype}, {size}, alignment=128)"
            if workspace is None
            else f"chain_c_workspace + {workspace.region(prefix).byte_offset // 4}"
        )
        lines.extend(
            [
                f"{prefix} = cute.make_tensor({pointer}, {layout})",
                f"for {prefix}_step in cutlass.range_constexpr({(size + plan.threads - 1) // plan.threads}):",
                f"    {prefix}_offset = chain_thread + {prefix}_step * {plan.threads}",
                f"    if {prefix}_offset < {size}:",
                _indent(expression.lines, 8),
                f"        {prefix}[{', '.join(coords)}] = {dtype}({value})",
            ]
        )
    lines.append("cute.arch.sync_threads()")
    return lines


def advance_carries(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    boundaries: dict[Node, str],
    scratch: ScratchLayouts | None = None,
    *,
    execution: ChainedExecution | None = None,
    resident_carries: frozenset[int] = frozenset(),
) -> list[str]:
    """Evaluate every next carry before overwriting any current carry element."""
    from .chained_matmul import _Expression
    from .chained_matmul import _indent
    from .chained_matmul import _shape

    loop = plan.loop
    assert loop is not None
    execution = execution or ChainedExecution(plan.threads)
    lines: list[str] = []
    stores: list[str] = []
    for carry in loop.region.carries:
        if carry.input_index in resident_carries:
            continue
        shape = _shape(carry.input)
        size = math.prod(shape)
        steps = (size + execution.threads - 1) // execution.threads
        prefix = loop.carry_name(carry.input_index)
        dtype = CompileEnvironment.current().backend.dtype_str(
            carry.input.meta["val"].dtype
        )
        if (
            plan.loop_workspace is not None
            and scratch is not None
            and carry.output in plan.dots
            and _shape(carry.output) == shape
        ):
            result = f"chain_{plan.dots.index(carry.output)}_c"
            current = plan.loop_workspace.region(prefix)
            published = plan.loop_workspace.region(result)
            if (
                boundaries.get(carry.output) == result
                and current.byte_offset == published.byte_offset
                and current.byte_size == published.byte_size
                and scratch.layout(prefix, shape) == scratch.layout(result, shape)
            ):
                # The final typed value already occupies this exact carry view.
                # Other carries have disjoint initialized slots, so their later
                # writeback cannot clobber it. The transition still joins the
                # complete role even when every carry is already in place.
                continue
        coords = (f"{prefix}_offset // {shape[1]}", f"{prefix}_offset % {shape[1]}")
        expression = _Expression(cg, plan, boundaries)
        expression.coordinate_names.add(f"{prefix}_offset")
        value = expression.value(carry.output, coords)
        lines.extend(
            [
                f"{prefix}_next = cute.make_rmem_tensor(({steps},), {dtype})",
                f"for {prefix}_step in cutlass.range_constexpr({steps}):",
                f"    {prefix}_offset = {execution.thread} + {prefix}_step * {execution.threads}",
                f"    if {prefix}_offset < {size}:",
                _indent(expression.lines, 8),
                f"        {prefix}_next[{prefix}_step] = {dtype}({value})",
            ]
        )
        stores.extend(
            [
                f"for {prefix}_step in cutlass.range_constexpr({steps}):",
                f"    {prefix}_offset = {execution.thread} + {prefix}_step * {execution.threads}",
                f"    if {prefix}_offset < {size}:",
                f"        {prefix}[{', '.join(coords)}] = {prefix}_next[{prefix}_step]",
            ]
        )
    if not stores:
        return [*lines, execution.sync]
    return [*lines, execution.sync, *stores, execution.sync]
