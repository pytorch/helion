"""Certify a pure consumer of the two halves of a staged row projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy
import torch
from torch.fx import Node

from ....language import _tracing_ops as tr
from ....language import memory_ops
from ....language import tile_ops
from ...device_ir import codegen_node_order
from ..promote_output_axis import _fresh_tensors
from .indexed_bf16_rows import _call
from .indexed_bf16_rows import _load
from .indexed_row_phases import IndexedRowPhase
from .indexed_row_phases import _axis
from .indexed_row_phases import _closed
from .indexed_row_phases import _indices
from .indexed_row_phases import _separate_destinations
from .indexed_row_phases import _tensor
from .match_common import host_tensor_name
from .match_common import node_tensor

if TYPE_CHECKING:
    from ...compile_environment import CompileEnvironment
    from ...device_ir import DeviceIR


@dataclass(frozen=True)
class RowEpilogue:
    consumer: int
    input: str
    output: str
    columns: int


def _pure(value: object, loads: tuple[Node, Node]) -> bool:
    # FX is acyclic but the dependency chain can exceed Python's recursion
    # limit. Seed discovery must not prevent an otherwise valid kernel binding.
    pending = [value]
    seen: set[Node] = set()
    while pending:
        node = pending.pop()
        if type(node) in (float, int):
            continue
        if not isinstance(node, Node):
            return False
        if node in loads or node in seen:
            continue
        if node.kwargs or node.op != "call_function":
            return False
        op = node.target
        if op is torch.ops.prims.convert_element_type.default:
            if len(node.args) != 2 or node.args[1] not in (
                torch.float32,
                torch.bfloat16,
            ):
                return False
            args = node.args[:1]
        elif (
            op is torch.ops.aten.tanh.default
            and len(node.args) == 1
            or op
            in (
                torch.ops.aten.add.Tensor,
                torch.ops.aten.mul.Tensor,
                torch.ops.aten.sub.Tensor,
            )
            and len(node.args) == 2
        ):
            args = node.args
        else:
            return False
        seen.add(node)
        pending.extend(args)
    return True


def indexed_row_epilogue(
    env: CompileEnvironment, ir: DeviceIR, source: IndexedRowPhase
) -> RowEpilogue | None:
    """Two separate consecutive phases; every (row, column) is independent.

    Keep the producer's stores. Other consumers of those stores, including a
    returned view, remain ordered by the join after this pair. Moving the
    destination write earlier must not destroy a concurrent producer input:
    check physical storage, including views captured in nested control flow.
    """
    if not source.paired or ir.host_function is None:
        return None
    pos = ir.root_ids.index(source.root)
    if pos + 1 >= len(ir.root_ids):
        return None
    phase = ir.phase_for_root(pos)
    if (
        # This schedule shares the following join. A terminal consumer
        # needs a different completion proof and retains its ordinary option.
        phase + 2 >= len(ir.phases)
        or ir.phases[phase].roots != [pos]
        or ir.phases[phase + 1].roots != [pos + 1]
    ):
        return None
    root = ir.root_ids[pos + 1]
    family = ir.task_families[pos + 1]
    if len(family.axes) != 2 or not all(a.canonical_origin for a in family.axes):
        return None
    extents = tuple(
        int(a.extent) for a in family.axes if isinstance(a.extent, (int, sympy.Integer))
    )
    if len(extents) != 2:
        return None
    rows, columns = extents
    if rows <= 0 or columns != source.rows:
        return None
    graph = ir.graphs[root].graph
    stores = [n for n in graph.nodes if _call(n, memory_ops.store, 4)]
    reads = [n for n in graph.nodes if _load(n, torch.bfloat16, 2)]
    if len(stores) != 1 or len(reads) != 2 or not _closed(graph, stores):
        return None
    (store,) = stores
    dst_node, _, expr, extra = store.args
    ix = _indices(store)
    if (
        extra is not None
        or len(ix) != 2
        or tuple(_axis(env, a) for a in ix) != family.logical_axis_order
        or any(_call(a, tr._get_symnode, 1) is None for a in ix)
        or (dest := host_tensor_name(dst_node)) is None
    ):
        return None
    low, high = reads if _indices(reads[0]) == ix else tuple(reversed(reads))
    off = _indices(high)
    if (
        _indices(low) != ix
        or len(off) != 2
        or off[0] is not ix[0]
        or (add := _call(off[1], torch.ops.aten.add.Tensor, 2)) is None
        or type(add.args[1]) is not int
        or add.args[1] != columns
        or (index := _call(add.args[0], tile_ops.tile_index, 1)) is None
        or index.args != (ix[1],)
        or low.args[0] is not high.args[0]
        or host_tensor_name(low.args[0]) != source.output
        or not _pure(expr, (low, high))
    ):
        return None
    gate = _tensor(low.args[0], (rows, 2 * columns), torch.bfloat16)
    out = _tensor(dst_node, (rows, columns), torch.bfloat16)
    if gate is None or out is None:
        return None
    all_inputs = []
    actual_gate = False
    for _, n in codegen_node_order(ir.graphs, [source.root], unreached=False):
        if (name := host_tensor_name(n)) is None:
            continue
        t = node_tensor(n)
        if t is None:
            return None
        all_inputs.append(t)
        if name == source.output:
            if t is not gate:
                return None
            actual_gate = True
    if not actual_gate or not _separate_destinations(
        env, _fresh_tensors(ir.host_function), all_inputs, [out]
    ):
        return None
    return RowEpilogue(root, source.output, dest, columns)
