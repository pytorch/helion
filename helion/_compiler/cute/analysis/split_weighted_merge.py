"""Proof of a bounded two-pass, masked softmax-weighted split reduction."""

from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.fx import Node

from ....language import _tracing_ops as ops
from ....language import creation_ops
from ....language import memory_ops
from ....language import tile_ops
from ....language import view_ops
from ...device_ir import ForLoopGraphInfo
from ...device_ir import IfGraphInfo
from ..register_loads import host_storages_may_alias
from .match_common import MatchFailure as _Mismatch
from .match_common import input_closure
from .match_common import is_call
from .match_common import match_args as _args
from .match_common import node_tensor
from .match_common import require_match as _require
from .match_common import strip_new_var

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...compile_environment import CompileEnvironment
    from ...device_ir import GraphInfo


@dataclass(frozen=True)
class SplitWeightedMerge:
    live: Node
    scores: Node
    values: Node
    output: Node
    split_axis: int


def _outputs(g: torch.fx.Graph) -> list[Node]:
    stmt = g.find_nodes(op="output")[0]
    seq = stmt.args[0]
    _require(isinstance(seq, (tuple, list)))
    return list(cast("list[Node]", seq))


def _covered(g: torch.fx.Graph, sinks: list[Node]) -> None:
    # Do not erase a sibling effect, even when its result never reaches the
    # visible carries (stores, inline assembly, nested barriers, etc.).
    active = set().union(*(input_closure(s) for s in sinks))
    _require(all(n.op in ("output", "placeholder") or n in active for n in g.nodes))


def _axis(x: object, env: CompileEnvironment) -> int:
    _require(
        isinstance(x, Node)
        and is_call(x, ops._get_symnode, torch.ops.aten.sym_size.int)
    )
    val = cast("Node", x).meta.get("val")
    _require(isinstance(val, torch.SymInt))
    bid = env.resolve_block_id(cast("torch.SymInt", val))
    _require(type(bid) is int)
    return cast("int", bid)


def _axes(x: object, env: CompileEnvironment) -> list[int]:
    _require(isinstance(x, (list, tuple)))
    return [_axis(a, env) for a in cast("list[object]", x)]


def _typed(x: object, dtype: torch.dtype, rank: int) -> torch.Tensor:
    t = node_tensor(x)
    _require(t is not None and t.dtype == dtype and t.ndim == rank)
    return cast("torch.Tensor", t)


def _scalar(x: object, value: float) -> None:
    _require(is_call(x, torch.ops.aten.scalar_tensor.default))
    n = cast("Node", x)
    _require(n.args == (value,) and not set(n.kwargs) - {"dtype", "device", "layout"})
    _typed(n, torch.float32, 0)


def _view(x: object, index: Sequence[object]) -> object:
    src, subscripts = _args(x, view_ops.subscript, 2)
    _require(subscripts == index)
    return src


def _loop(
    call: Node, graphs: Sequence[GraphInfo], env: CompileEnvironment, carries: int
) -> tuple[int, list[Node], list[Node], list[Node]]:
    gid, start, stop, captures = _args(call, ops._for_loop, 4)
    _require(type(gid) is int and start == [0] and stop == [64])
    _require(0 <= cast("int", gid) < len(graphs))
    info = graphs[cast("int", gid)]
    _require(isinstance(info, ForLoopGraphInfo))
    body = cast("ForLoopGraphInfo", info)
    _require(len(body.block_ids) == 1 and not body.needs_barrier_before)
    (axis,) = body.block_ids
    params = body.graph.find_nodes(op="placeholder")
    _require(
        isinstance(captures, (tuple, list))
        and len(params) == len(cast("list[Node]", captures)) == 2 * carries
    )
    conditionals = body.graph.find_nodes(op="call_function", target=ops._if)
    _require(len(conditionals) == 1)
    (cond,) = conditionals
    pred, then_id, else_id, args, no_args = _args(cond, ops._if, 5)
    _require(type(then_id) is int and type(else_id) is int and no_args == [])
    _require(
        isinstance(args, (tuple, list))
        and [strip_new_var(a) for a in cast("list[Node]", args)] == params
    )
    limit, begin = _args(pred, torch.ops.aten.gt.Scalar, 2)
    (axis_sym,) = _args(begin, tile_ops.tile_begin, 1)
    _require(strip_new_var(limit) is params[0] and _axis(axis_sym, env) == axis)
    _require(all(0 <= cast("int", gid) < len(graphs) for gid in (then_id, else_id)))
    then = graphs[cast("int", then_id)]
    other = graphs[cast("int", else_id)]
    _require(isinstance(then, IfGraphInfo) and not _outputs(other.graph))
    _covered(other.graph, [])
    branch = cast("IfGraphInfo", then)
    _require(branch.else_branch == else_id and branch.predicate_is_tensor is False)
    joined = _outputs(body.graph)
    _require(len(joined) == carries and len(branch.branches_outputs or ()) == carries)
    _covered(body.graph, joined)
    for i, result in enumerate(joined):
        prev, updated = _args(result, ops._phi, 2)
        _require(_args(prev, operator.getitem, 2) == (cond, i + carries))
        _require(_args(updated, operator.getitem, 2) == (cond, i))
    names = branch.if_arg_names
    _require(
        names is not None
        and len(names) == len(params)
        and len(set(names)) == len(params)
    )
    _require(branch.else_arg_names == [])
    _require(
        branch.branches_outputs
        == [
            (i, cast("list[str]", names)[len(params) - carries + i])
            for i in range(carries)
        ]
    )
    inputs = branch.graph.find_nodes(op="placeholder")
    _require(len(inputs) == len(params))
    outputs = _outputs(branch.graph)[:carries]
    _require(len(outputs) == carries)
    _covered(branch.graph, outputs)
    return axis, list(cast("list[Node]", captures)), inputs, outputs


def prove_split_weighted_merge(
    graph: torch.fx.Graph,
    graphs: Sequence[GraphInfo],
    env: CompileEnvironment,
    row: int,
    column: int,
) -> SplitWeightedMerge | None:
    """Prove axis, carry, address, mask and alias identity through both loops.

    Only the scalar live bound is opaque, and it is compiled with the normal
    graph lowering. Its input loads are scalar and CTA-uniform. Sixty-four
    split slots are the only measured two-elements-per-warp-lane schedule.
    """
    aten = torch.ops.aten
    sl = slice(None)
    negative_infinity = -float("inf")
    try:
        loops = graph.find_nodes(op="call_function", target=ops._for_loop)
        stores = graph.find_nodes(op="call_function", target=memory_ops.store)
        _require(len(loops) == 2 and len(stores) == 1)
        lmax, lsum = loops
        (store,) = stores
        dst, idx, stored, mask = _args(store, memory_ops.store, 4)
        _require(
            mask is None
            and _axes(idx, env) == [row, column]
            and is_call(dst, ops._host_tensor)
        )
        _require(graph.find_nodes(op="output")[0].args == (None,))
        _covered(graph, [store])
        sa, c0, p0, max_out = _loop(lmax, graphs, env, 1)
        sb, c1, p1, sum_out = _loop(lsum, graphs, env, 2)
        _require(sa == sb and len({sa, row, column}) == 3 and c0[0] is c1[0])
        live = c0[0]
        _typed(live, torch.int32, 0)
        # Live may be any expression composed of these scalar, uniform ops.
        # Do not hoist per-tile/per-lane inputs or an unrelated effect.
        srcs = []
        for n in input_closure(live):
            if is_call(n, ops._host_tensor):
                continue
            _typed(n, torch.int32, 0)
            if is_call(n, memory_ops.load):
                host, coords, guard, fallback = _args(n, memory_ops.load, 4)
                _require(
                    is_call(host, ops._host_tensor)
                    and guard is None
                    and fallback is None
                )
                _require(
                    isinstance(coords, (tuple, list))
                    and all(type(i) is int for i in cast("list[object]", coords))
                )
                srcs.append(cast("Node", host))
            else:
                _require(
                    is_call(
                        n,
                        aten.add.Tensor,
                        aten.sub.Tensor,
                        aten.mul.Tensor,
                        aten.div.Tensor_mode,
                    )
                )

        def full(x: object, dimensions: list[int], value: float) -> None:
            shape, fill, dt, extra = _args(x, creation_ops.full, 4)
            _require(
                _axes(shape, env) == dimensions
                and fill == value
                and dt == torch.float32
                and extra is None
            )

        full(c0[1], [row], negative_infinity)
        full(c1[2], [row], 0)
        full(c1[3], [row, column], 0)
        eq, zero, top = _args(c1[1], aten.where.self, 3)
        _require(_args(eq, aten.eq.Scalar, 2) == (top, negative_infinity))
        _scalar(zero, 0)

        def joined(x: object, loop: Node, index: int, init: Node) -> None:
            old, extracted = _args(x, ops._phi, 2)
            _require(
                old is init and _args(extracted, operator.getitem, 2) == (loop, index)
            )

        joined(top, lmax, 0, c0[1])
        val, dt = _args(stored, torch.ops.prims.convert_element_type.default, 2)
        _require(dt == torch.bfloat16)
        numerator, denom = _args(val, aten.div.Tensor, 2)
        joined(numerator, lsum, 1, c1[3])
        joined(_view(denom, [sl, None]), lsum, 0, c1[2])

        def masked_load(
            x: object, dims: list[int], provided: object, arg: Node
        ) -> Node:
            host, index, masked, mode = _args(x, memory_ops.load, 4)
            _require(
                is_call(host, ops._host_tensor)
                and mode is None
                and _axes(index, env) == dims
            )
            valid = _view(masked, [sl] + [None] * (len(dims) - 1))
            ids, limit = _args(valid, aten.lt.Tensor, 2)
            (split,) = _args(ids, tile_ops.tile_index, 1)
            _require(_axis(split, env) == sa and strip_new_var(limit) is arg)
            if provided is not None:
                _require(valid is provided)
            return cast("Node", host)

        old, group = _args(max_out[0], aten.maximum.default, 2)
        _require(strip_new_var(old) is p0[1])
        masked, axis = _args(group, aten.amax.default, 2)
        chosen, identity = _args(masked, ops._mask_to, 2)
        _require(axis == [0] and identity == negative_infinity)
        cond, lse, neg_inf = _args(chosen, aten.where.self, 3)
        _scalar(neg_inf, negative_infinity)
        scores = masked_load(lse, [sa, row], _view(cond, [sl, None]), p0[0])

        def summed(x: Node, previous: Node, *, product: bool = False) -> object:
            prev, total = _args(x, aten.add.Tensor, 2)
            _require(strip_new_var(prev) is previous)
            mask_to, which = _args(total, aten.sum.dim_IntList, 2)
            _require(which == [0])
            if is_call(mask_to, ops._mask_to):
                expr, neutral = _args(mask_to, ops._mask_to, 2)
                _require(neutral == 0)
                return expr
            # Fast math can drop the implicit zero mask around the numerator
            # product. Below, prove the weight is zero for every inactive
            # split and the value load has the identical live split mask.
            # The emitter requires full rows/columns and a unit split tile.
            _require(
                product and env.settings.fast_math and is_call(mask_to, aten.mul.Tensor)
            )
            return mask_to

        weight = summed(sum_out[0], p1[2])
        condition, e, zero2 = _args(weight, aten.where.self, 3)
        _scalar(zero2, 0)
        valid = _view(condition, [sl, None])
        (diff,) = _args(e, aten.exp2.default, 1)
        lse2, best2 = _args(diff, aten.sub.Tensor, 2)
        _require(strip_new_var(_view(best2, [None, sl])) is p1[1])
        same_scores = masked_load(lse2, [sa, row], valid, p1[0])
        _require(
            scores.args == same_scores.args
            and node_tensor(scores) is node_tensor(same_scores)
        )
        w, partial = _args(summed(sum_out[1], p1[3], product=True), aten.mul.Tensor, 2)
        _require(_view(w, [sl, sl, None]) is weight)
        values = masked_load(partial, [sa, row, column], valid, p1[0])
        dst_tensor = _typed(dst, torch.bfloat16, 2)
        lse_tensor = _typed(scores, torch.float32, 2)
        val_tensor = _typed(values, torch.float32, 3)
        rows, cols = dst_tensor.shape
        _require(
            type(rows) is int
            and type(cols) is int
            and rows > 0
            and cols > 0
            and cols % 128 == 0
        )
        row_extent, column_extent = (
            env.block_sizes[row].size,
            env.block_sizes[column].size,
        )
        if not isinstance(row_extent, (int, torch.SymInt)) or not isinstance(
            column_extent, (int, torch.SymInt)
        ):
            return None
        _require(
            env.known_equal(row_extent, rows) and env.known_equal(column_extent, cols)
        )
        _require(
            tuple(lse_tensor.shape) == (64, rows)
            and tuple(val_tensor.shape) == (64, rows, cols)
        )
        _require(
            env.traced_strides_hold(dst_tensor)
            and all(type(s) is int for s in dst_tensor.stride())
        )
        # Disjoint rows/columns suffice; transposed outputs also qualify.
        rs, cs = dst_tensor.stride()
        _require(rs > 0 and cs > 0 and (rs >= cols * cs or cs >= rows * rs))
        for source in [scores, values, *srcs]:
            src = node_tensor(source)
            _require(
                src is not None and not host_storages_may_alias(env, dst_tensor, src)
            )
        return SplitWeightedMerge(live, scores, values, cast("Node", dst), sa)
    except _Mismatch:
        return None
