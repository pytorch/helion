"""Prove the enclosing stores and scalar inputs of an indexed BF16 row sum."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import starmap
from math import prod
import operator
from typing import TYPE_CHECKING

import sympy
import torch
from torch.fx import Graph
from torch.fx import Node

from ....language import _tracing_ops as tr
from ....language import memory_ops
from ....language import tile_ops
from ...runtime_tensor_guards import tensors_have_proven_disjoint_storage
from ..fragment_vector_io import static_int
from ..fragment_vector_io import static_strides
from ..promote_output_axis import _fresh_tensors
from .indexed_bf16_rows import SelectedRows
from .indexed_bf16_rows import _call
from .indexed_bf16_rows import _cast
from .indexed_bf16_rows import _load
from .indexed_bf16_rows import indexed_row_sums
from .match_common import host_tensor_name
from .match_common import node_tensor
from .match_common import unit_fixed_block

if TYPE_CHECKING:
    from ...compile_environment import CompileEnvironment
    from ...device_ir import DeviceIR


@dataclass(frozen=True)
class IndexedRowPhase:
    root: int
    row_axis: int
    k_axis: int
    rows: int
    k: int
    groups: int
    instances: int  # Two outer unit-tile coordinates of the staged root.
    # Nonzero only for a singleton input in its own dependency phase.
    invariant_rows: int
    matrix: str
    vector: str
    selection: str
    weights: str
    output: str
    input_scale: str | None = None
    selected_ids: str | None = None
    selected_weights: str | None = None

    @property
    def paired(self) -> bool:
        return self.input_scale is not None


def _indices(node: Node) -> tuple[object, ...]:
    items = node.args[1]
    return tuple(items) if isinstance(items, (list, tuple)) else ()


def _axis(env: CompileEnvironment, expr: object) -> int | None:
    if not (
        _call(expr, tr._get_symnode, 1) or _call(expr, torch.ops.aten.sym_size.int, 2)
    ):
        return None
    assert isinstance(expr, Node)
    value = expr.meta.get("val")
    return env.resolve_block_id(value) if isinstance(value, torch.SymInt) else None


def _symbol(node: object) -> sympy.Basic | None:
    value = node.meta.get("val") if isinstance(node, Node) else None
    return value.node.expr if isinstance(value, torch.SymInt) else None


def _captured(captured: tuple[object, ...], outer: tuple[object, ...]) -> bool:
    def same(value: object, parent: object) -> bool:
        # Tracing may capture the outer scalar or emit tile.begin of the
        # same outer block inside the K loop.
        if _call(value, tr._get_symnode, 1):
            return _symbol(value) is not None and _symbol(value) == _symbol(parent)
        begin = _call(value, tile_ops.tile_begin, 1)
        origin = _call(parent, tile_ops.tile_begin, 1)
        return (
            begin is not None
            and origin is not None
            and _call(begin.args[0], tr._get_symnode, 1) is not None
            and _call(origin.args[0], tr._get_symnode, 1) is not None
            and _symbol(begin.args[0]) is not None
            and _symbol(begin.args[0]) == _symbol(origin.args[0])
        )

    return len(captured) == len(outer) and all(
        starmap(same, zip(captured, outer, strict=True))
    )


def _closed(graph: Graph, effects: list[Node]) -> bool:
    todo = effects.copy()
    live: set[Node] = set()
    while todo:
        n = todo.pop()
        if n not in live:
            live.add(n)
            todo.extend(n.all_input_nodes)
    return all(n in live or n.op == "output" for n in graph.nodes)


def _tensor(
    value: object, shape: tuple[int, ...], dtype: torch.dtype | None = None
) -> torch.Tensor | None:
    t = node_tensor(value)
    if (
        host_tensor_name(value) is None
        or t is None
        or dtype is not None
        and t.dtype is not dtype
        or tuple(map(static_int, t.shape)) != shape
    ):
        return None
    return t


def _separate_destinations(
    env: CompileEnvironment,
    fresh: set[torch.Tensor],
    reads: list[torch.Tensor],
    writes: list[torch.Tensor],
) -> bool:
    # A fresh allocation alone can have overlapping rows (empty_strided).
    for idx, t in enumerate(writes):
        stride = static_strides(env, t)
        if t not in fresh or stride is None:
            return False
        expected = 1
        for size, s in zip(reversed(t.shape), reversed(stride), strict=True):
            dim = static_int(size)
            if dim is None or dim <= 0 or dim > 1 and s != expected:
                return False
            expected *= dim
        if expected >= 2**31:
            return False
        # A host conditional can have fresh synthetic storage while either
        # source aliases an output. The grid join can move only if every
        # possible read and write origin remains disjoint.
        if any(
            not tensors_have_proven_disjoint_storage(env, t, other)
            for other in (*writes[:idx], *reads)
        ):
            return False
    for t in reads:
        strides = static_strides(env, t)
        if strides is None:
            return False
        span = 0
        for size, stride in zip(t.shape, strides, strict=True):
            dim = static_int(size)
            if dim is None or dim < 1 or stride < 0:
                return False
            span += (dim - 1) * stride
        if span >= 2**31:
            return False
    return True


def _normalization(sums: SelectedRows, batch: Node) -> bool:
    inv = _call(sums.normalization, torch.ops.aten.rsqrt.default, 1)
    add = _call(inv.args[0], torch.ops.aten.add.Tensor, 2) if inv else None
    if add is None:
        return False
    mean, eps = add.args
    if not (
        _call(eps, tr._get_symnode, 1)
        and isinstance(eps, Node)
        and isinstance(eps.meta.get("val"), torch.SymFloat)
        and isinstance(mean, Node)
        and mean.target is torch.ops.aten.mean.dim
        and mean.args in ((None, [-1]), (None, (-1,)))
        and set(mean.kwargs) == {"_extra_args"}
    ):
        return False
    arg = mean.kwargs["_extra_args"]
    if (
        not isinstance(arg, (list, tuple))
        or len(arg) != 1
        or (ev := _call(arg[0], tr._inductor_lowering_extra, 1)) is None
    ):
        return False
    args = ev.args[0]
    if (
        not isinstance(args, (list, tuple))
        or len(args) != 1
        or (square := _call(args[0], torch.ops.aten.mul.Tensor, 2)) is None
        or square.args[0] is not square.args[1]
    ):
        return False
    raw = _cast(square.args[0], torch.float32)
    return (
        raw is not None
        and _load(raw, torch.bfloat16, 1) is not None
        and host_tensor_name(raw.args[0]) == host_tensor_name(sums.vector.args[0])
        and _indices(raw) == (batch, slice(None))
    )


def _publications(
    ir: DeviceIR, u: Node, sums: SelectedRows, axis: object, where: tuple[object, ...]
) -> tuple[Node, Node, Node] | None:
    if _call(u, tr._if, 5) is None:
        return None
    check, then, otherwise, captures, returns = u.args
    pred = _call(check, operator.eq, 2)
    if (
        pred is None
        or pred.args[1] != 0
        or (begin := _call(pred.args[0], tile_ops.tile_begin, 1)) is None
        or begin.args != (axis,)
        or captures not in ([sums.route], (sums.route,))
        or returns
        or type(then) is not int
        or type(otherwise) is not int
        or not 0 <= then < len(ir.graphs)
        or not 0 <= otherwise < len(ir.graphs)
    ):
        return None
    true = ir.graphs[then].graph
    if any(n.op != "output" for n in ir.graphs[otherwise].graph.nodes):
        return None
    stores = [n for n in true.nodes if _call(n, memory_ops.store, 4)]
    captures_ = list(true.find_nodes(op="placeholder"))
    if len(stores) != 2 or len(captures_) != 1 or not _closed(true, stores):
        return None
    ids, score = stores
    value = _call(ids.args[2], tr._new_var, 1)
    read = _load(score.args[2], torch.float32, 0)
    if (
        any(s.args[3] is not None or not _captured(_indices(s), where) for s in stores)
        or value is None
        or value.args != (captures_[0],)
        or read is None
        or _indices(read) != _indices(score)
    ):
        return None
    id_base, score_base, input_base = ids.args[0], score.args[0], read.args[0]
    if (
        isinstance(id_base, Node)
        and isinstance(score_base, Node)
        and isinstance(input_base, Node)
    ):
        return id_base, score_base, input_base
    return None


def _phase(
    env: CompileEnvironment,
    ir: DeviceIR,
    root: int,
    axis_ids: tuple[int, ...],
    sizes: tuple[int, ...],
    fresh: set[torch.Tensor],
) -> IndexedRowPhase | None:
    graph = ir.graphs[root].graph
    loops = [n for n in graph.nodes if _call(n, tr._for_loop, 4)]
    if len(loops) != 1 or (s := indexed_row_sums(env, ir, loops[0])) is None:
        return None
    B, S, R = sizes
    K = s.k
    if (
        len(axis_ids) != 3
        or _axis(env, s.row.args[0]) != axis_ids[2]
        or len(s.outputs) * R != s.bank_stride
    ):
        return None
    where = _indices(s.route)
    if len(where) != 2 or any(
        (b := _call(x, tile_ops.tile_begin, 1)) is None or _axis(env, b.args[0]) != axis
        for x, axis in zip(where, axis_ids[:2], strict=True)
    ):
        return None
    select = _tensor(s.route.args[0], (B, S))
    mat = node_tensor(s.matrix)
    if (
        select is None
        or mat is None
        or mat.ndim != 2
        or (height := static_int(mat.shape[0])) is None
        or not 0 < height < 2**31
        or height % s.bank_stride
        or static_int(mat.shape[1]) != K
        or prod((height, K)) >= 2**31
    ):
        return None
    # Rebased, contiguous K reads in both MMA fragments. The emitted fast
    # branch also checks runtime pointer alignment/strides on each binding.
    strides = static_strides(env, mat)
    if strides is None or strides[1] != 1 or strides[0] < K:
        return None
    stores = [n for n in graph.nodes if _call(n, memory_ops.store, 4)]
    branches = [n for n in graph.nodes if _call(n, tr._if, 5)]
    if (
        not stores
        or any(z.args[3] is not None for z in stores)
        or not _closed(graph, stores + branches)
    ):
        return None
    dst = stores[0].args[0]
    matrix, vector, selection, output = (
        host_tensor_name(n) for n in (s.matrix, s.vector.args[0], s.route.args[0], dst)
    )
    if matrix is None or vector is None or selection is None or output is None:
        return None
    reads = [mat, select]
    writes = []
    scale = id_out = score_out = None
    if len(s.outputs) == 1:
        if (
            len(stores) != 1
            or branches
            or s.normalization is not None
            or not _captured(_indices(s.vector)[:-1], where)
        ):
            return None
        ix = _indices(stores[0])
        vec = _tensor(s.vector.args[0], (B, S, K), torch.bfloat16)
        out = _tensor(dst, (B, S, R), torch.bfloat16)
        val = _cast(stores[0].args[2], torch.bfloat16)
        mul = _call(val, torch.ops.aten.mul.Tensor, 2)
        if (
            len(ix) != 3
            or ix[:2] != where
            or _axis(env, ix[-1]) != axis_ids[2]
            or mul is None
            or mul.args[0] is not s.outputs[0]
            or vec is None
            or out is None
        ):
            return None
        w = _load(mul.args[1], torch.float32, 0)
        if w is None or _indices(w) != where:
            return None
        score = w.args[0]
        reads.append(vec)
        writes.append(out)
    else:
        if (
            len(stores) != 2
            or len(branches) != 1
            or s.scale is None
            or not _captured(_indices(s.vector)[:-1], where[:1])
            or not isinstance(where[0], Node)
            or not _normalization(s, where[0])
        ):
            return None
        ix = _indices(stores[0])
        iy = _indices(stores[1])
        if (
            len(ix) != 2
            or len(iy) != 2
            or _axis(env, ix[1]) != axis_ids[2]
            or iy[0] is not ix[0]
        ):
            return None
        linear = _call(ix[0], operator.add, 2)
        high = _call(iy[1], torch.ops.aten.add.Tensor, 2)
        if (
            linear is None
            or linear.args[1] is not where[1]
            or (base := _call(linear.args[0], operator.mul, 2)) is None
            or base.args != (where[0], S)
            or high is None
            or high.args[1] != R
            or (tile := _call(high.args[0], tile_ops.tile_index, 1)) is None
            or tile.args != (ix[1],)
        ):
            return None
        if any(
            z.args[0] is not dst or _cast(z.args[2], torch.bfloat16) is not phi
            for z, phi in zip(stores, s.outputs, strict=True)
        ):
            return None
        pub = _publications(ir, branches[0], s, ix[1], where)
        if pub is None:
            return None
        ids, weights, score = pub
        kk = _tensor(s.scale.args[0], (K,), torch.bfloat16)
        x = _tensor(s.vector.args[0], (B, K), torch.bfloat16)
        out = _tensor(dst, (B * S, 2 * R), torch.bfloat16)
        ii = _tensor(ids, (B, S), select.dtype)
        ww = _tensor(weights, (B, S), torch.float32)
        if kk is None or x is None or out is None or ii is None or ww is None:
            return None
        reads.extend((kk, x))
        writes.extend((out, ii, ww))
        scale = host_tensor_name(s.scale.args[0])
        id_out = host_tensor_name(ids)
        score_out = host_tensor_name(weights)
    score_t = _tensor(score, (B, S), torch.float32)
    if (
        score_t is None
        or (score_name := host_tensor_name(score)) is None
        or not _separate_destinations(env, fresh, [*reads, score_t], writes)
    ):
        return None
    invariant_rows = 0
    if scale is not None and B == 1:
        pos = ir.root_ids.index(root)
        if ir.phases[ir.phase_for_root(pos)].roots == [pos]:
            invariant_rows = S * R
    return IndexedRowPhase(
        root,
        axis_ids[2],
        s.k_axis,
        R,
        K,
        height // s.bank_stride,
        B * S,
        invariant_rows,
        matrix,
        vector,
        selection,
        score_name,
        output,
        scale,
        id_out,
        score_out,
    )


def indexed_row_phases(
    env: CompileEnvironment, ir: DeviceIR
) -> tuple[IndexedRowPhase, ...]:
    if (
        # BF16 MMA may round differently from the original widened FP32 sum.
        not env.settings.fast_math
        or env.process_group_name is not None
        or ir.host_function is None
        or env.config_spec.target_device_capability is None
        or env.config_spec.target_device_capability[0] < 8
    ):
        return ()
    fresh = None
    matches = []
    for root, fam in zip(ir.root_ids, ir.task_families, strict=True):
        sizes = tuple(
            int(a.extent)
            for a in fam.axes
            if a.canonical_origin and isinstance(a.extent, (int, sympy.Integer))
        )
        if (
            len(fam.axes) != 3
            or len(sizes) != 3
            or min(sizes) <= 0
            or not all(unit_fixed_block(env, a.block_id) for a in fam.axes[:2])
        ):
            continue
        if fresh is None:
            fresh = _fresh_tensors(ir.host_function)
        if (
            f := _phase(env, ir, root, fam.logical_axis_order, sizes, fresh)
        ) is not None:
            matches.append(f)
    return tuple(matches)
