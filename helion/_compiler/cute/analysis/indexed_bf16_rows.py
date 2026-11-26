"""Recognize complete FP32 sums of BF16 rows selected by a shared integer ID.

This is a graph witness, not a rewrite. In particular, parent stores, scalar
publication and the physical CTA layout need separate proofs before replacing
even a structurally eligible loop.
"""

from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import TYPE_CHECKING

import torch
from torch.fx import Node

from ....language import _tracing_ops as tr
from ....language import creation_ops
from ....language import memory_ops
from ....language import tile_ops
from ...device_ir import ForLoopGraphInfo
from ..fragment_vector_io import static_int
from .match_common import host_tensor_name
from .match_common import node_tensor

if TYPE_CHECKING:
    from ...compile_environment import CompileEnvironment
    from ...device_ir import DeviceIR


@dataclass(frozen=True)
class SelectedRows:
    loop: Node
    matrix: Node
    row: Node
    vector: Node
    route: Node
    bank_stride: int
    k_axis: int
    k: int
    outputs: tuple[Node, ...]
    normalization: Node | None
    scale: Node | None


def _call(value: object, target: object, arity: int) -> Node | None:
    if (
        isinstance(value, Node)
        and value.op == "call_function"
        and value.target is target
        and not value.kwargs
        and len(value.args) == arity
    ):
        return value
    return None


def _cast(value: object, dtype: torch.dtype) -> Node | None:
    node = _call(value, torch.ops.prims.convert_element_type.default, 2)
    if node is not None and node.args[1] is dtype and isinstance(node.args[0], Node):
        return node.args[0]
    return None


def _load(value: object, dtype: torch.dtype, rank: int) -> Node | None:
    node = _call(value, memory_ops.load, 4)
    if (
        node is not None
        and node.args[2:] == (None, None)
        and host_tensor_name(node.args[0]) is not None
        and isinstance(node.args[1], (list, tuple))
        and (t := node_tensor(node)) is not None
        and t.dtype is dtype
        and t.ndim == rank
    ):
        return node
    return None


def _capture(value: object, captures: dict[Node, Node]) -> Node | None:
    var = _call(value, tr._new_var, 1)
    if var is not None and isinstance(var.args[0], Node):
        return captures.get(var.args[0])
    return None


def _only_user(value: Node) -> Node | None:
    return next(iter(value.users)) if len(value.users) == 1 else None


def _complete_graph(inner: ForLoopGraphInfo, output: Node) -> bool:
    live: set[Node] = set()
    todo = [output]
    while todo:
        n = todo.pop()
        if n not in live:
            live.add(n)
            todo.extend(n.all_input_nodes)
    return all(n in live for n in inner.graph.nodes)


def indexed_row_sums(
    env: CompileEnvironment, ir: DeviceIR, call: Node
) -> SelectedRows | None:
    """Bind the true loop-K symbol, independent carries, and their consuming phis."""
    if _call(call, tr._for_loop, 4) is None:
        return None
    gid, begins, ends, args = call.args
    if (
        type(gid) is not int
        or not 0 <= gid < len(ir.graphs)
        or begins not in ((0,), [0])
        or not isinstance(ends, (list, tuple))
        or len(ends) != 1
        or type(k := ends[0]) is not int
        or not 0 < k < 2**31
        or not isinstance(args, (tuple, list))
    ):
        return None
    inner = ir.graphs[gid]
    if not isinstance(inner, ForLoopGraphInfo) or len(inner.block_ids) != 1:
        return None
    (k_axis,) = inner.block_ids
    ph = list(inner.graph.find_nodes(op="placeholder"))
    if len(ph) != len(args):
        return None
    captures = {p: a for p, a in zip(ph, args, strict=True) if isinstance(a, Node)}
    (output,) = inner.graph.find_nodes(op="output")
    results = output.args[0]
    if (
        not isinstance(results, (tuple, list))
        or len(results) not in (1, 2)
        or len(results) != len(call.users)
        or not _complete_graph(inner, output)
    ):
        return None

    def full_window(n: Node, ix: Node) -> bool:
        tensor = node_tensor(n.args[0])
        indices = n.args[1]
        # Physical input layout is rechecked by the generated wrapper; reject
        # incomplete K windows and any symbol other than the inner K tile.
        return (
            tensor is not None
            and tensor.ndim > 0
            and static_int(tensor.shape[-1]) == k
            and isinstance(indices, (tuple, list))
            and len(indices) == tensor.ndim
            and indices[-1] is ix
            and isinstance(size := ix.meta.get("val"), torch.SymInt)
            and env.resolve_block_id(size) == k_axis
        )

    accumulators: set[Node] = set()
    phis: list[Node] = []
    rows: list[Node] = []
    operands: list[tuple[Node, Node, Node]] = []
    for pos, result in enumerate(results):
        update = _call(result, torch.ops.aten.add.Tensor, 2)
        if update is None:
            return None
        initial = _capture(update.args[0], captures)
        if (
            initial is None
            or _call(initial, creation_ops.full, 4) is None
            or initial.args[1:] != (0.0, torch.float32, None)
            or initial in accumulators
            or (it := node_tensor(initial)) is None
            or it.ndim != 1
        ):
            return None
        accumulators.add(initial)
        items = [
            n
            for n in call.users
            if _call(n, operator.getitem, 2) and n.args == (call, pos)
        ]
        if len(items) != 1 or (phi := _only_user(items[0])) is None:
            return None
        if _call(phi, tr._phi, 2) is None or phi.args != (initial, items[0]):
            return None
        phis.append(phi)
        red = _call(update.args[1], torch.ops.aten.sum.dim_IntList, 2)
        if red is None or red.args[1] not in ((-1,), [-1]):
            return None
        operand = red.args[0]
        mask = _call(operand, tr._mask_to, 2)
        if mask is not None:
            if mask.args[1] != 0:
                return None
            operand = mask.args[0]
        elif not env.settings.fast_math:
            return None
        # Relaxed math may omit its neutral padding. The selected transform
        # below still requires a plain mul of the two proven loads at the
        # same K symbol. Emission covers complete physical K/row panels.
        product = _call(operand, torch.ops.aten.mul.Tensor, 2)
        if product is None:
            return None
        widened = [_cast(n, torch.float32) for n in product.args]
        matrices = [n for n in widened if _load(n, torch.bfloat16, 2) is not None]
        vectors = [
            n
            for n in widened
            if n is not None
            and (t := node_tensor(n)) is not None
            and t.ndim == 1
            and t.dtype is torch.bfloat16
        ]
        if len(matrices) != 1 or len(vectors) != 1 or matrices[0] is None:
            return None
        matrix, vector = matrices[0], vectors[0]
        host, indices = matrix.args[:2]
        if (
            not isinstance(host, Node)
            or not isinstance(indices, (list, tuple))
            or len(indices) != 2
        ):
            return None
        row = _capture(indices[0], captures)
        idx = _call(indices[1], tr._get_symnode, 1)
        if row is None or idx is None or not full_window(matrix, idx):
            return None
        rows.append(row)
        operands.append((host, idx, vector))
    if len(set(operands)) != 1:
        return None
    matrix, kidx, vector = operands[0]
    row = rows[0]
    if _call(row, torch.ops.aten.add.Tensor, 2) is None:
        return None
    group = _call(row.args[0], torch.ops.aten.mul.Tensor, 2)
    row_tile = _call(row.args[1], tile_ops.tile_index, 1)
    if group is None or row_tile is None:
        return None
    selection, stride = group.args
    route = _load(selection, torch.int32, 0) or _load(selection, torch.int64, 0)
    if route is None or type(stride) is not int or not 0 < stride < 2**31:
        return None
    if len(rows) == 2 and (
        stride % 2
        or _call(rows[1], torch.ops.aten.add.Tensor, 2) is None
        or rows[1].args != (row, stride // 2)
    ):
        return None
    norm = scale = None
    if _load(vector, torch.bfloat16, 1) is None:
        # Preserve the two separate BF16 roundings in
        # BF16(FP32(vector) * FP32(normalization)) * BF16(scale).
        if _call(vector, torch.ops.aten.mul.Tensor, 2) is None:
            return None
        rounded = _cast(vector.args[0], torch.bfloat16)
        scale = _load(vector.args[1], torch.bfloat16, 1)
        if (
            rounded is None
            or _call(rounded, torch.ops.aten.mul.Tensor, 2) is None
            or scale is None
        ):
            return None
        norm = _capture(rounded.args[1], captures)
        raw = _cast(rounded.args[0], torch.float32)
        if (
            norm is None
            or (t := node_tensor(norm)) is None
            or t.ndim != 0
            or t.dtype is not torch.float32
            or raw is None
            or _load(raw, torch.bfloat16, 1) is None
            or not full_window(scale, kidx)
        ):
            return None
        vector = raw
    if not full_window(vector, kidx):
        return None
    return SelectedRows(
        call,
        matrix,
        row_tile,
        vector,
        route,
        stride,
        k_axis,
        k,
        tuple(phis),
        norm,
        scale,
    )
