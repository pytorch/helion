"""Assert the authored page recurrence on its actual lowered graphs.

This is a CPU fixture contract, not a compiler admission predicate. Address,
physical head/CTA ownership and a dedicated attention schedule still require
the general matcher plan. No graph or metadata is rewritten here.
"""

from __future__ import annotations

import math
import operator
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.fx import Node

from helion._compiler.cute.analysis.match_common import symnode_sympy
from helion._compiler.device_ir import ForLoopGraphInfo
from helion._compiler.device_ir import IfGraphInfo
from helion._compiler.device_ir import NodeArgsGraphInfo
from helion._compiler.device_ir import RootGraphInfo
from helion._compiler.device_ir import control_flow_child_graph_ids
from helion._compiler.variable_origin import GridOrigin
from helion.language import _tracing_ops as tracing
from helion.language import creation_ops
from helion.language import inline_asm_ops
from helion.language import matmul_ops
from helion.language import memory_ops
from helion.language import view_ops

if TYPE_CHECKING:
    from helion._compiler.device_ir import DeviceIR


def assert_conditional_page_contract(ir: DeviceIR) -> dict[str, object]:
    """Check captures, phi pairings, whole-matrix vote and FP32/BF16 edges."""
    infos = {info.graph: info for info in ir.graphs}
    roots = [info for info in ir.graphs if isinstance(info, RootGraphInfo)]
    assert len(roots) == 1
    visited: set[int] = set()

    def visit(graph_id: int) -> None:
        assert graph_id not in visited
        visited.add(graph_id)
        info = ir.graphs[graph_id]
        assert info.graph_id == graph_id
        seen: set[Node] = set()
        for node in info.graph.nodes:
            assert all(source in seen for source in node.all_input_nodes)
            seen.add(node)
            children = control_flow_child_graph_ids(node)
            if children:
                assert node.target is tracing._if or tracing.is_for_loop_target(
                    node.target
                )
                slots = (3, 4) if node.target is tracing._if else (3,)
                for child, slot in zip(children, slots, strict=True):
                    nested = ir.graphs[child]
                    assert isinstance(nested, NodeArgsGraphInfo)
                    actual = node.args[slot]
                    assert isinstance(actual, (tuple, list))
                    assert list(nested.node_args) == list(actual)
                    assert len(nested.graph.find_nodes(op="placeholder")) == len(
                        nested.node_args
                    )
                    visit(child)

    visit(roots[0].graph_id)
    assert visited == set(range(len(ir.graphs)))
    loops = [info for info in ir.graphs if isinstance(info, ForLoopGraphInfo)]
    assert len(loops) == 1
    loop = loops[0]

    def resolve(value: object) -> Node:
        assert isinstance(value, Node)
        while True:
            if value.target is tracing._new_var:
                value = value.args[0]
                assert isinstance(value, Node)
            elif value.op == "placeholder" and value.graph is not loop.graph:
                info = infos[value.graph]
                assert isinstance(info, NodeArgsGraphInfo)
                value = info.placeholder_to_outer_arg(value)
            else:
                return value

    def args(value: object, target: object) -> tuple:
        node = resolve(value)
        assert node.target is target and not node.kwargs, node
        return node.args

    def same(left: object, right: object) -> None:
        assert resolve(left) is resolve(right)

    def shape(value: object, dtype: torch.dtype) -> tuple[int, ...]:
        tensor = resolve(value).meta["val"]
        assert isinstance(tensor, torch.Tensor) and tensor.dtype == dtype
        assert all(type(size) is int for size in tensor.shape)
        return tuple(tensor.shape)

    def scalar(value: object, expected: float) -> None:
        node = resolve(value)
        assert node.target is torch.ops.aten.scalar_tensor.default
        assert node.args == (expected,) and shape(node, torch.float32) == ()

    def join(value: object) -> tuple[Node, Node, Node]:
        false, true = args(value, tracing._phi)
        call, index = args(true, operator.getitem)
        false_call, false_index = args(false, operator.getitem)
        assert call is false_call and isinstance(call, Node)
        assert call.target is tracing._if
        info = ir.graphs[cast("int", call.args[1])]
        assert isinstance(info, IfGraphInfo) and info.branches_outputs is not None
        count = len(info.branches_outputs)
        assert false_index == index + count
        captures: dict[str, Node] = {}
        for names, actual in zip(
            (info.if_arg_names, info.else_arg_names), call.args[3:5], strict=True
        ):
            assert names is not None
            assert isinstance(actual, (tuple, list))
            for name, source in zip(names, actual, strict=True):
                assert isinstance(source, Node)
                assert name not in captures or captures[name] is source
                captures[name] = source
        values = []
        for side, slot in enumerate(info.branches_outputs[index]):
            branch = ir.graphs[cast("int", call.args[side + 1])]
            output = branch.graph.find_nodes(op="output")[0].args[0]
            values.append(output[slot] if type(slot) is int else captures[slot])
        return call, resolve(values[0]), resolve(values[1])

    dots = [node for node in loop.graph.nodes if node.target is matmul_ops.dot]
    assert len(dots) == 2
    qk, pv = dots
    rows, keys = shape(qk, torch.float32)
    assert qk.args[2:] == (None, None)
    maximum, output, denominator = loop.graph.find_nodes(op="output")[0].args[0]
    assert output is pv
    vote_call, new_maximum, old_maximum = join(maximum)
    selected_l, selected_p = args(denominator, torch.ops.aten.add.Tensor)
    l_call, scaled_l, old_l = join(selected_l)
    p_call, new_p, old_p = join(selected_p)
    assert l_call is p_call is vote_call
    cast_p, value, selected_o, output_dtype = pv.args
    assert output_dtype is None
    p, dtype = args(cast_p, torch.ops.prims.convert_element_type.default)
    assert dtype is torch.bfloat16
    same(p, selected_p)
    o_call, scaled_o, old_o = join(selected_o)
    assert o_call is vote_call
    assert shape(old_maximum, torch.float32) == (rows,)
    assert shape(old_l, torch.float32) == (rows, keys)
    assert shape(old_o, torch.float32) == shape(pv, torch.float32)
    assert shape(value, torch.bfloat16)[0] == keys

    def centered(value: object, maximum: Node) -> tuple[Node, float]:
        difference = resolve(args(value, torch.ops.aten.exp2.default)[0])
        asm, constraints, operands, dtype, pure, pack = args(
            difference, inline_asm_ops.inline_asm_elementwise
        )
        assert (asm, constraints, dtype, pure, pack) == (
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            torch.float32,
            True,
            1,
        )
        score, factor, additive = operands
        same(score, qk)
        factor = resolve(factor)
        assert factor.target is torch.ops.aten.full.default
        assert isinstance(factor.args[0], (tuple, list))
        assert tuple(factor.args[0]) == (rows, keys)
        assert shape(factor, torch.float32) == (rows, keys)
        coefficient = factor.args[1]
        assert type(coefficient) is float and math.isfinite(coefficient)
        visible, negative, masked = args(additive, torch.ops.aten.where.self)
        scalar(masked, -math.inf)
        expanded = args(negative, torch.ops.aten.neg.default)[0]
        source, axis = args(expanded, torch.ops.aten.unsqueeze.default)
        assert axis == 1
        same(source, maximum)
        assert shape(visible, torch.bool) == (rows, keys)
        return resolve(visible), coefficient

    visible, coefficient = centered(old_p, old_maximum)
    updated_visible, updated_coefficient = centered(new_p, new_maximum)
    assert visible is updated_visible and coefficient == updated_coefficient
    difference = args(old_p, torch.ops.aten.exp2.default)[0]
    vote, dtype = args(vote_call.args[0], torch.ops.prims.convert_element_type.default)
    assert dtype is torch.bool and shape(vote, torch.uint8) == ()
    for axis in (0, 1):
        masked, axes = args(vote, torch.ops.aten.amax.default)
        assert axes == [axis]
        vote, identity = args(masked, tracing._mask_to)
        assert identity == 0
    comparison, dtype = args(vote, torch.ops.prims.convert_element_type.default)
    assert dtype is torch.uint8
    compared, threshold = args(comparison, torch.ops.aten.gt.Scalar)
    same(compared, difference)
    assert threshold == 2.0 and shape(comparison, torch.bool) == (rows, keys)

    previous, reduced = args(new_maximum, torch.ops.aten.fmax.default)
    same(previous, old_maximum)
    maximum_scores, axes = args(reduced, torch.ops.aten.amax.default)
    assert axes == [1]
    maximum_scores, identity = args(maximum_scores, tracing._mask_to)
    assert identity == -math.inf
    isnan, neutral, scaled_scores = args(maximum_scores, torch.ops.aten.where.self)
    same(args(isnan, torch.ops.aten.isnan.default)[0], scaled_scores)
    scalar(neutral, -math.inf)
    mask, scaled_scores, neutral = args(scaled_scores, torch.ops.aten.where.self)
    same(mask, visible)
    scalar(neutral, -math.inf)
    score, scale = args(scaled_scores, torch.ops.aten.mul.Tensor)
    same(score, qk)
    assert scale == coefficient

    rescale_o, updated_o, identity_o = join(scaled_o)
    rescale_l, updated_l, identity_l = join(scaled_l)
    assert rescale_o is rescale_l
    same(identity_o, old_o)
    same(identity_l, old_l)
    alpha = None
    for updated, original in ((updated_o, old_o), (updated_l, old_l)):
        source, expanded = args(updated, torch.ops.aten.mul.Tensor)
        same(source, original)
        current, slices = args(expanded, view_ops.subscript)
        assert slices == [slice(None), None]
        if alpha is None:
            alpha = resolve(current)
        else:
            same(current, alpha)
    test, one, exponential = args(alpha, torch.ops.aten.where.self)
    tested, sentinel = args(test, torch.ops.aten.eq.Scalar)
    same(tested, new_maximum)
    assert sentinel == -math.inf
    scalar(one, 1.0)
    delta = args(exponential, torch.ops.aten.exp2.default)[0]
    previous, current = args(delta, torch.ops.aten.sub.Tensor)
    same(previous, old_maximum)
    same(current, new_maximum)
    iteration, zero = args(rescale_o.args[0], operator.gt)
    assert resolve(iteration).target is tracing._get_symnode and zero == 0
    assert ir.host_function is not None
    symbol = symnode_sympy(resolve(iteration))
    assert symbol is not None
    origin = ir.host_function.expr_to_origin[symbol].origin
    assert isinstance(origin, GridOrigin) and loop.block_ids == [origin.block_id]

    loop_call = next(
        node
        for info in ir.graphs
        for node in info.graph.nodes
        if tracing.is_for_loop_target(node.target) and node.args[0] == loop.graph_id
    )
    placeholders = list(loop.graph.find_nodes(op="placeholder"))
    assert len({old_maximum, old_o, old_l}) == 3
    outer_state = []
    for index, (state, fill) in enumerate(
        ((old_maximum, -math.inf), (old_o, 0.0), (old_l, 0.0))
    ):
        initial = loop.node_args[placeholders.index(state)]
        sizes, value, dtype, layout = args(initial, creation_ops.full)
        assert tuple(sizes) == shape(state, torch.float32)
        assert value == fill and dtype is torch.float32 and layout is None
        if fill == 0:
            assert math.copysign(1, value) == 1
        extracted = next(
            node
            for node in loop_call.users
            if node.target is operator.getitem and node.args[1] == index
        )
        users = list(extracted.users)
        assert len(users) == 1
        assert args(users[0], tracing._phi) == (initial, extracted)
        outer_state.append(users[0])

    stores = [
        node
        for info in ir.graphs
        for node in info.graph.nodes
        if node.target is memory_ops.store
    ]
    assert len(stores) == 3
    written = {node.args[2] for node in stores}
    assert outer_state[0] in written and outer_state[1] in written
    final_l = (written - {outer_state[0], outer_state[1]}).pop()
    # Reverse the exact four-group left fold, then the 16/8/4/2/1 lane tree.
    pieces = []
    for group in (3, 2, 1):
        final_l, selected = args(final_l, torch.ops.aten.add.Tensor)
        source, axis, index = args(selected, torch.ops.aten.select.int)
        assert axis == 1 and index == group
        pieces.append(resolve(source))
    source, axis, index = args(final_l, torch.ops.aten.select.int)
    assert axis == 1 and index == 0
    assert all(piece is resolve(source) for piece in pieces)
    left, right = args(source, torch.ops.aten.add.Tensor)
    previous, axis, index = args(left, torch.ops.aten.select.int)
    peer, peer_axis, peer_index = args(right, torch.ops.aten.select.int)
    same(previous, peer)
    assert (axis, index, peer_axis, peer_index) == (2, 0, 2, 1)
    for width in (2, 4, 8, 16):
        left, right = args(previous, torch.ops.aten.add.Tensor)
        previous, axis, start, end = args(left, torch.ops.aten.slice.Tensor)
        peer, peer_axis, peer_start, peer_end = args(right, torch.ops.aten.slice.Tensor)
        same(previous, peer)
        assert (axis, start, end) == (2, 0, width)
        assert (peer_axis, peer_start, peer_end) == (2, width, 2**63 - 1)
    lanes, sizes = args(previous, torch.ops.aten.view.default)
    assert sizes == [rows, 4, 32]
    same(lanes, outer_state[2])

    return {
        "root_graph": roots[0].graph_id,
        "page_graph": loop.graph_id,
        "vote_graph": vote_call.args[1],
        "rescale_graph": rescale_o.args[1],
        "rows": rows,
        "keys": keys,
        "vote_elements": rows * keys,
        "threshold": threshold,
        "factor": coefficient,
        "zero_iteration_initials": ["-inf", "+0", "+0"],
        "denominator_tree": [16, 8, 4, 2, 1],
        "denominator_group_fold": "((0+1)+2)+3",
        "scope": "exact first-root logical page recurrence; no transport admission",
    }
