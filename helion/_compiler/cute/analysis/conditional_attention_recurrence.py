"""Semantic facts for conditional, per-key online attention.

This parser owns branch/carry arithmetic only. Native operand semantics,
address masks, aliasing, physical vote ownership and an emitter are separate
obligations; finding this recurrence never enables a generic MMA route.
"""

from __future__ import annotations

import dataclasses
import math
import operator
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.fx import Node

from ....language import _tracing_ops as tracing
from ....language import creation_ops
from ....language import inline_asm_ops
from ....language import matmul_ops
from ....language import memory_ops
from ....language import view_ops
from ...device_ir import ForLoopGraphInfo
from ...device_ir import IfGraphInfo
from ...device_ir import NodeArgsGraphInfo
from ...device_ir import RootGraphInfo
from ...variable_origin import GridOrigin
from .conditional_continuation import ConditionalContinuation
from .conditional_continuation import prove_conditional_continuation
from .conditional_denominator import require_denominator_tree
from .conditional_regions import ConditionalRegions
from .conditional_regions import prove_conditional_regions
from .match_common import symnode_sympy

if TYPE_CHECKING:
    from ...compile_environment import CompileEnvironment
    from ...device_ir import DeviceIR


@dataclasses.dataclass(frozen=True)
class ConditionalAttentionRecurrence:
    regions: ConditionalRegions
    loop: ForLoopGraphInfo
    qk: Node
    pv: Node
    maximum_input: Node
    accumulator_input: Node
    denominator_input: Node
    probability: Node
    threshold_branch: Node
    rescale_branch: Node
    visible: Node
    score_factor: float
    threshold: float
    loop_outputs: tuple[Node, ...]
    stores: tuple[Node, ...]
    final_denominator: Node
    proof_nodes: frozenset[Node]
    continuation: ConditionalContinuation | None = None


class _Mismatch(Exception):
    pass


def _read_recurrence(
    ir: DeviceIR,
    regions: ConditionalRegions,
    continuation: ConditionalContinuation | None = None,
) -> ConditionalAttentionRecurrence:
    """Check captures, phi pairings, whole-matrix vote and FP32/BF16 edges."""
    infos = {region.graph.graph: region.graph for region in regions.regions}
    roots = [info for info in infos.values() if isinstance(info, RootGraphInfo)]
    if not len(roots) == 1:
        raise _Mismatch
    loops = [info for info in infos.values() if isinstance(info, ForLoopGraphInfo)]
    if not len(loops) == 1:
        raise _Mismatch
    loop = loops[0]
    proof_nodes: set[Node] = set()

    def resolve(value: object) -> Node:
        if not isinstance(value, Node):
            raise _Mismatch
        while True:
            proof_nodes.add(value)
            if value.target is tracing._new_var:
                value = value.args[0]
                if not isinstance(value, Node):
                    raise _Mismatch
            elif value.op == "placeholder" and value.graph is not loop.graph:
                info = infos[value.graph]
                if not isinstance(info, NodeArgsGraphInfo):
                    raise _Mismatch
                value = info.placeholder_to_outer_arg(value)
            else:
                return value

    def args(value: object, target: object, arity: int = 2) -> tuple:
        node = resolve(value)
        if not (node.target is target and len(node.args) == arity and not node.kwargs):
            raise _Mismatch
        return node.args

    def same(left: object, right: object) -> None:
        if resolve(left) is not resolve(right):
            raise _Mismatch

    def require(condition: bool) -> None:
        if not condition:
            raise _Mismatch

    def shape(value: object, dtype: torch.dtype) -> tuple[int, ...]:
        tensor = resolve(value).meta["val"]
        if not (isinstance(tensor, torch.Tensor) and tensor.dtype == dtype):
            raise _Mismatch
        if not all(type(size) is int for size in tensor.shape):
            raise _Mismatch
        return tuple(tensor.shape)

    def scalar(value: object, expected: float) -> None:
        node = resolve(value)
        if node.target is not torch.ops.aten.scalar_tensor.default:
            raise _Mismatch
        if not (node.args == (expected,) and shape(node, torch.float32) == ()):
            raise _Mismatch

    def join(value: object) -> tuple[Node, Node, Node]:
        false, true = args(value, tracing._phi)
        call, index = args(true, operator.getitem)
        false_call, false_index = args(false, operator.getitem)
        if not (call is false_call and isinstance(call, Node)):
            raise _Mismatch
        if call.target is not tracing._if:
            raise _Mismatch
        info = ir.graphs[cast("int", call.args[1])]
        if not (isinstance(info, IfGraphInfo) and info.branches_outputs is not None):
            raise _Mismatch
        count = len(info.branches_outputs)
        if not false_index == index + count:
            raise _Mismatch
        captures: dict[str, Node] = {}
        for names, actual in zip(
            (info.if_arg_names, info.else_arg_names), call.args[3:5], strict=True
        ):
            if not names is not None:
                raise _Mismatch
            if not isinstance(actual, (tuple, list)):
                raise _Mismatch
            for name, source in zip(names, actual, strict=True):
                if not isinstance(source, Node):
                    raise _Mismatch
                if not (name not in captures or captures[name] is source):
                    raise _Mismatch
                captures[name] = source
        values = []
        for side, slot in enumerate(info.branches_outputs[index]):
            branch = ir.graphs[cast("int", call.args[side + 1])]
            output = branch.graph.find_nodes(op="output")[0].args[0]
            values.append(output[slot] if type(slot) is int else captures[slot])
        return (call, resolve(values[0]), resolve(values[1]))

    dots = [node for node in loop.graph.nodes if node.target is matmul_ops.dot]
    if not len(dots) == 2:
        raise _Mismatch
    qk, pv = dots
    dimensions = shape(qk, torch.float32)
    if len(dimensions) != 2:
        raise _Mismatch
    rows, keys = dimensions
    if not qk.args[2:] == (None, None):
        raise _Mismatch
    outputs = loop.graph.find_nodes(op="output")[0].args[0]
    if len(outputs) != 3:
        raise _Mismatch
    maximum, output, denominator = outputs
    if output is not pv:
        raise _Mismatch
    vote_call, new_maximum, old_maximum = join(maximum)
    selected_l, selected_p = args(denominator, torch.ops.aten.add.Tensor)
    l_call, scaled_l, old_l = join(selected_l)
    p_call, new_p, old_p = join(selected_p)
    if not l_call is p_call is vote_call:
        raise _Mismatch
    cast_p, value, selected_o, output_dtype = args(pv, matmul_ops.dot, 4)
    if output_dtype is not None:
        raise _Mismatch
    p, dtype = args(cast_p, torch.ops.prims.convert_element_type.default)
    if dtype is not torch.bfloat16:
        raise _Mismatch
    same(p, selected_p)
    o_call, scaled_o, old_o = join(selected_o)
    if o_call is not vote_call:
        raise _Mismatch
    if not shape(old_maximum, torch.float32) == (rows,):
        raise _Mismatch
    if not shape(old_l, torch.float32) == (rows, keys):
        raise _Mismatch
    if not shape(old_o, torch.float32) == shape(pv, torch.float32):
        raise _Mismatch
    if not shape(value, torch.bfloat16)[0] == keys:
        raise _Mismatch

    def centered(value: object, maximum: Node) -> tuple[Node, float]:
        difference = resolve(args(value, torch.ops.aten.exp2.default, 1)[0])
        asm, constraints, operands, dtype, pure, pack = args(
            difference, inline_asm_ops.inline_asm_elementwise, 6
        )
        if not (asm, constraints, dtype, pure, pack) == (
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            torch.float32,
            True,
            1,
        ):
            raise _Mismatch
        if not isinstance(operands, (tuple, list)) or len(operands) != 3:
            raise _Mismatch
        score, factor, additive = operands
        same(score, qk)
        factor = resolve(factor)
        if factor.target is not torch.ops.aten.full.default:
            raise _Mismatch
        if not isinstance(factor.args[0], (tuple, list)):
            raise _Mismatch
        if not tuple(factor.args[0]) == (rows, keys):
            raise _Mismatch
        if not shape(factor, torch.float32) == (rows, keys):
            raise _Mismatch
        coefficient = factor.args[1]
        if not (type(coefficient) is float and math.isfinite(coefficient)):
            raise _Mismatch
        visible, negative, masked = args(additive, torch.ops.aten.where.self, 3)
        scalar(masked, -math.inf)
        expanded = args(negative, torch.ops.aten.neg.default, 1)[0]
        source, axis = args(expanded, torch.ops.aten.unsqueeze.default)
        if not axis == 1:
            raise _Mismatch
        same(source, maximum)
        if not shape(visible, torch.bool) == (rows, keys):
            raise _Mismatch
        return (resolve(visible), coefficient)

    visible, coefficient = centered(old_p, old_maximum)
    updated_visible, updated_coefficient = centered(new_p, new_maximum)
    if not (visible is updated_visible and coefficient == updated_coefficient):
        raise _Mismatch
    difference = args(old_p, torch.ops.aten.exp2.default, 1)[0]
    vote, dtype = args(vote_call.args[0], torch.ops.prims.convert_element_type.default)
    if not (dtype is torch.bool and shape(vote, torch.uint8) == ()):
        raise _Mismatch
    for axis in (0, 1):
        masked, axes = args(vote, torch.ops.aten.amax.default)
        if not axes == [axis]:
            raise _Mismatch
        vote, identity = args(masked, tracing._mask_to)
        if not identity == 0:
            raise _Mismatch
    comparison, dtype = args(vote, torch.ops.prims.convert_element_type.default)
    if dtype is not torch.uint8:
        raise _Mismatch
    compared, threshold = args(comparison, torch.ops.aten.gt.Scalar)
    same(compared, difference)
    if not (
        (type(threshold) is float and math.isfinite(threshold) and (threshold > 0))
        and shape(comparison, torch.bool) == (rows, keys)
    ):
        raise _Mismatch
    previous, reduced = args(new_maximum, torch.ops.aten.fmax.default)
    same(previous, old_maximum)
    maximum_scores, axes = args(reduced, torch.ops.aten.amax.default)
    if not axes == [1]:
        raise _Mismatch
    maximum_scores, identity = args(maximum_scores, tracing._mask_to)
    if not identity == -math.inf:
        raise _Mismatch
    isnan, neutral, scaled_scores = args(maximum_scores, torch.ops.aten.where.self, 3)
    same(args(isnan, torch.ops.aten.isnan.default, 1)[0], scaled_scores)
    scalar(neutral, -math.inf)
    mask, scaled_scores, neutral = args(scaled_scores, torch.ops.aten.where.self, 3)
    same(mask, visible)
    scalar(neutral, -math.inf)
    score, scale = args(scaled_scores, torch.ops.aten.mul.Tensor)
    same(score, qk)
    if not scale == coefficient:
        raise _Mismatch
    rescale_o, updated_o, identity_o = join(scaled_o)
    rescale_l, updated_l, identity_l = join(scaled_l)
    if rescale_o is not rescale_l:
        raise _Mismatch
    same(identity_o, old_o)
    same(identity_l, old_l)
    alpha = None
    for updated, original in ((updated_o, old_o), (updated_l, old_l)):
        source, expanded = args(updated, torch.ops.aten.mul.Tensor)
        same(source, original)
        current, slices = args(expanded, view_ops.subscript)
        if not slices == [slice(None), None]:
            raise _Mismatch
        if alpha is None:
            alpha = resolve(current)
        else:
            same(current, alpha)
    test, one, exponential = args(alpha, torch.ops.aten.where.self, 3)
    tested, sentinel = args(test, torch.ops.aten.eq.Scalar)
    same(tested, new_maximum)
    if not sentinel == -math.inf:
        raise _Mismatch
    scalar(one, 1.0)
    delta = args(exponential, torch.ops.aten.exp2.default, 1)[0]
    previous, current = args(delta, torch.ops.aten.sub.Tensor)
    same(previous, old_maximum)
    same(current, new_maximum)
    if continuation is not None:
        if (
            continuation.loop is not loop
            or continuation.rescale_branch is not rescale_o
        ):
            raise _Mismatch
        proof_nodes.update(continuation.proof_nodes)
    else:
        iteration, zero = args(rescale_o.args[0], operator.gt)
        if not (resolve(iteration).target is tracing._get_symnode and zero == 0):
            raise _Mismatch
        if not ir.host_function is not None:
            raise _Mismatch
        symbol = symnode_sympy(resolve(iteration))
        if not symbol is not None:
            raise _Mismatch
        tracked = ir.host_function.expr_to_origin.get(symbol)
        if tracked is None:
            raise _Mismatch
        origin = tracked.origin
        if not (isinstance(origin, GridOrigin) and loop.block_ids == [origin.block_id]):
            raise _Mismatch
    loop_call = next(
        node
        for info in ir.graphs
        for node in info.graph.nodes
        if tracing.is_for_loop_target(node.target) and node.args[0] == loop.graph_id
    )
    placeholders = list(loop.graph.find_nodes(op="placeholder"))
    if not len({old_maximum, old_o, old_l}) == 3:
        raise _Mismatch
    outer_state = []
    for index, (state, fill) in enumerate(
        ((old_maximum, -math.inf), (old_o, 0.0), (old_l, 0.0))
    ):
        if state not in placeholders:
            raise _Mismatch
        initial = loop.node_args[placeholders.index(state)]
        if continuation is not None:
            if initial is not continuation.initial_states[index]:
                raise _Mismatch
        else:
            sizes, value, dtype, layout = args(initial, creation_ops.full, 4)
            if not tuple(sizes) == shape(state, torch.float32):
                raise _Mismatch
            if not (value == fill and dtype is torch.float32 and (layout is None)):
                raise _Mismatch
            if fill == 0:
                if not math.copysign(1, value) == 1:
                    raise _Mismatch
        extracted = next(
            node
            for node in loop_call.users
            if node.target is operator.getitem and node.args[1] == index
        )
        users = list(extracted.users)
        if not len(users) == 1:
            raise _Mismatch
        if not args(users[0], tracing._phi) == (initial, extracted):
            raise _Mismatch
        outer_state.append(users[0])
    stores = [
        node
        for info in ir.graphs
        for node in info.graph.nodes
        if node.target is memory_ops.store
    ]
    if continuation is not None:
        if tuple(outer_state) != continuation.loop_outputs or tuple(stores) != (
            continuation.output_store,
            continuation.lse_store,
        ):
            raise _Mismatch
        final_l = continuation.denominator
    else:
        if not len(stores) == 3:
            raise _Mismatch
        written = {node.args[2] for node in stores}
        if len(written) != 3:
            raise _Mismatch
        if not (outer_state[0] in written and outer_state[1] in written):
            raise _Mismatch
        final_l = (written - {outer_state[0], outer_state[1]}).pop()
    final_denominator = final_l
    require_denominator_tree(
        final_l, outer_state[2], rows, call=args, same=same, require=require
    )
    return ConditionalAttentionRecurrence(
        regions,
        loop,
        qk,
        pv,
        old_maximum,
        old_o,
        old_l,
        selected_p,
        vote_call,
        rescale_o,
        visible,
        coefficient,
        threshold,
        tuple(outer_state),
        tuple(stores),
        final_denominator,
        frozenset(proof_nodes),
        continuation,
    )


def match_conditional_attention_recurrence(
    ir: DeviceIR, env: CompileEnvironment
) -> ConditionalAttentionRecurrence | None:
    if len(ir.root_ids) != 1:
        return None
    regions = prove_conditional_regions(ir.graphs, ir.root_ids[0])
    if regions is None:
        return None
    try:
        return _read_recurrence(ir, regions)
    except _Mismatch:
        continuation = prove_conditional_continuation(ir, env)
        if continuation is None:
            return None
        try:
            return _read_recurrence(ir, regions, continuation)
        except _Mismatch:
            return None
