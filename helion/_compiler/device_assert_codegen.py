"""Backend-neutral checks and codegen of ``hl.device_assert``."""

from __future__ import annotations

import ast
import operator
from typing import TYPE_CHECKING

import sympy
import torch
from torch.utils._sympy.functions import FloorDiv

from .. import exc
from ..language._tracing_ops import _if
from ..language._tracing_ops import _while_loop
from ..language._tracing_ops import is_for_loop_target
from ..language.debug_ops import device_assert
from .ast_extension import create
from .ast_extension import expr_from_string
from .compile_environment import CompileEnvironment
from .inductor_lowering_extra import padded_range_lengths
from .node_masking import loop_carries

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .device_ir import DeviceIR
    from .device_ir import GraphInfo
    from .inductor_lowering import CodegenState

_RESHAPES = (
    torch.ops.aten.view.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten._unsafe_view.default,
)
_REORDERS = (
    torch.ops.aten.permute.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.t.default,
)


def check_device_asserts(device_ir: DeviceIR) -> None:
    """Reject an ``hl.device_assert`` whose condition, followed into the
    loops and branches that hand it in, carry it or return it, comes from
    lanes the tile masks of its ``_mask_to`` do not describe: a reshape that
    moves a tile's lanes (``x[rows, cols].flatten()``, or
    ``x[rows, :].reshape(2, rows)``, whose rows dim holds other lanes than the
    rows tile's), an ``hl.arange`` whose length is computed from block sizes
    but not a power of two (``hl.arange(3 * tile.block_size)``, whose padded
    lanes are no constant count to mask), or a padded ``hl.arange`` whose
    dims a permute or a reshape rearranged: ``hl.arange(100)[:, None]`` is
    [128, 1], and the walk that finds its 28 padded lanes does not follow the
    permute; ``hl.arange(100).reshape(2, 50)`` puts lanes 100 to 113 in its
    second row.  The walk goes through reductions too: a reduction of lanes a
    reshape moved reads padding lanes no mask covers."""
    asserts = [
        node
        for info in device_ir.graphs
        for node in info.graph.find_nodes(op="call_function", target=device_assert)
    ]
    if not asserts:
        return
    env = CompileEnvironment.current()
    infos = {id(info.graph): info for info in device_ir.graphs}
    bodies = {info.graph_id: info.graph for info in device_ir.graphs}
    loops: dict[int, torch.fx.Node] = {}
    for info in device_ir.graphs:
        for node in info.graph.nodes:
            if is_for_loop_target(node.target):
                loops[node.args[0]] = node  # pyrefly: ignore[unsupported-operation]
            elif node.target is _while_loop:
                loops[node.args[1]] = node  # pyrefly: ignore[unsupported-operation]
    for node in asserts:
        with node.meta["location"]:
            condition = node.args[0]
            val = (
                condition.meta["val"] if isinstance(condition, torch.fx.Node) else None
            )
            if isinstance(val, torch.Tensor):
                sizes = val.size()
                padded = [s for s in sizes if _padded_constant(s)]
                if padded and any(env.get_block_id(s) is not None for s in sizes):
                    # torch.stack([v, v, v]) of a tile's v: Triton pads the 3
                    # to 4, which neither the tile masks (a broadcast to a
                    # shape of 3) nor the padded-lane mask (on the logical
                    # dims, which flatten_loops merges) describe.
                    raise exc.InvalidAPIUsage(
                        "hl.device_assert() of a condition with a constant dim of "
                        f"{padded[0]}, no power of two, next to a tile's dims: "
                        "reduce it first, as with .all(dim)"
                    )
            _check_condition(condition, env, infos, bodies, loops)


def _check_condition(
    condition: object,
    env: CompileEnvironment,
    infos: dict[int, GraphInfo],
    bodies: dict[int, torch.fx.Graph],
    loops: dict[int, torch.fx.Node],
) -> None:
    from .device_ir import NodeArgsGraphInfo

    seen: set[tuple[torch.fx.Node, bool]] = set()
    # Each value with whether a permute lies between it and the condition.
    stack: list[tuple[object, bool]] = [(condition, False)]
    while stack:
        current, reordered = stack.pop()
        if not isinstance(current, torch.fx.Node) or (current, reordered) in seen:
            continue
        seen.add((current, reordered))
        if current.op == "placeholder":
            owner = infos.get(id(current.graph))
            if isinstance(owner, NodeArgsGraphInfo):
                stack.append((owner.placeholder_to_outer_arg(current), reordered))
                loop = loops.get(owner.graph_id)
                if loop is not None:
                    stack.extend(
                        (carry.update, reordered)
                        for carry in loop_carries(loop, owner.graph)
                        if carry.placeholder is current
                    )
            continue
        flow = current.args[0] if current.target is operator.getitem else None
        if isinstance(flow, torch.fx.Node) and (
            is_for_loop_target(flow.target) or flow.target in (_while_loop, _if)
        ):
            # A loop's result is its body's output at the same index, not the
            # inputs of the other values it carries. A branch's results
            # interleave both branches' outputs with the values one leaves
            # alone (which the _phi after it reads): take every output.
            index = current.args[1]
            assert isinstance(index, int)
            if flow.target is _if:
                graph_ids = [flow.args[1], flow.args[2]]
            else:
                graph_ids = [flow.args[1 if flow.target is _while_loop else 0]]
            for graph_id in graph_ids:
                assert isinstance(graph_id, int)
                (output,) = bodies[graph_id].find_nodes(op="output")
                outputs = output.args[0]
                assert isinstance(outputs, (list, tuple))
                if flow.target is _if:
                    stack.extend((value, reordered) for value in outputs)
                else:
                    stack.append((outputs[index], reordered))
            continue
        if current.target in _RESHAPES and _moves_tile_lanes(
            current.args[0], current, env
        ):
            raise exc.InvalidAPIUsage(
                "hl.device_assert() of a condition reshaped from tiles, whose "
                "lanes the tile masks no longer find: assert before the reshape"
            )
        if current.target is torch.ops.prims.iota.default:
            length = current.args[0]
            if isinstance(length, torch.fx.Node):
                length = length.meta["val"]
            assert isinstance(length, (int, torch.SymInt))
            if not _whole_lanes(length, env):
                raise exc.InvalidAPIUsage(
                    "hl.device_assert() of a condition over an hl.arange of a "
                    f"computed length ({length}), whose padded lanes it cannot "
                    "tell apart: use a constant length or a block size"
                )
            if reordered and isinstance(length, int) and length & (length - 1):
                raise exc.InvalidAPIUsage(
                    "hl.device_assert() of a condition over a padded hl.arange "
                    "whose dims a permute or a reshape rearranged: assert before it"
                )
        reorders = (
            reordered
            or current.target in _REORDERS
            or (current.target in _RESHAPES and _changes_dims(current))
        )
        stack.extend((node, reorders) for node in current.all_input_nodes)


def _padded_constant(size: int | torch.SymInt) -> bool:
    """Whether a dim of ``size`` is a constant that is no power of two, which
    a Triton tensor pads to the next one."""
    return isinstance(size, int) and size > 1 and (size & (size - 1)) != 0


def _whole_lanes(length: int | torch.SymInt, env: CompileEnvironment) -> bool:
    """Whether an ``hl.arange`` of ``length`` has a constant count of padded
    lanes: a constant, or a power-of-two multiple of block sizes (each a
    power of two) and their power-of-two quotients."""
    if isinstance(length, int):
        return True
    expr = length.node.expr
    assert isinstance(expr, sympy.Expr)
    coefficient, rest = expr.as_coeff_Mul()
    return (
        isinstance(coefficient, sympy.Integer)
        and _is_power_of_two(coefficient)
        and all(_block_power(factor, env) for factor in sympy.Mul.make_args(rest))
    )


def _block_power(factor: sympy.Basic, env: CompileEnvironment) -> bool:
    if env.get_block_id(factor) is not None:
        return True
    return (
        isinstance(factor, FloorDiv)
        and env.get_block_id(factor.args[0]) is not None
        and isinstance(factor.args[1], sympy.Integer)
        and _is_power_of_two(factor.args[1])
    )


def _is_power_of_two(value: sympy.Integer) -> bool:
    return value > 0 and (int(value) & (int(value) - 1)) == 0


def _moves_tile_lanes(
    source: object, reshaped: torch.fx.Node, env: CompileEnvironment
) -> bool:
    """Whether the reshape of ``source`` into ``reshaped`` puts a tile's dim
    where it holds other lanes: it is kept when the same dim stands after
    as many elements and before as many as it did ([tm, 128] into
    [tm, 2, 64]); size-1 dims come and go."""
    assert isinstance(source, torch.fx.Node)
    before = _exprs(source.meta["val"].size())
    after = _exprs(reshaped.meta["val"].size())
    for i, size in enumerate(source.meta["val"].size()):
        if env.get_block_id(size) is None:
            continue
        outer, inner = sympy.Mul(*before[:i]), sympy.Mul(*before[i + 1 :])
        if not any(
            after[k] == before[i]
            and sympy.Mul(*after[:k]) == outer
            and sympy.Mul(*after[k + 1 :]) == inner
            for k in range(len(after))
        ):
            return True
    return False


def _changes_dims(reshaped: torch.fx.Node) -> bool:
    """Whether a reshape does more than add or drop size-1 dims."""
    source = reshaped.args[0]
    assert isinstance(source, torch.fx.Node)
    before = [e for e in _exprs(source.meta["val"].size()) if e != 1]
    after = [e for e in _exprs(reshaped.meta["val"].size()) if e != 1]
    return before != after


def _exprs(sizes: Sequence[int | torch.SymInt]) -> list[sympy.Basic]:
    # Compared as expressions: a SymInt == would guard on a block size.
    return [
        size.node.expr if isinstance(size, torch.SymInt) else sympy.Integer(size)
        for size in sizes
    ]


def emit_device_assert(state: CodegenState, function: str) -> None:
    """Emit the ``hl.device_assert`` of ``state`` as ``function(condition,
    message)`` (traced only with ``Settings.device_asserts``)."""
    condition = state.ast_arg(0)
    message = state.proxy_arg(1)
    assert isinstance(message, str)
    fake = state.proxy_arg(0)
    if isinstance(fake, torch.Tensor):
        if 0 in fake.size():
            # An empty condition holds; Triton would compute it as a scalar.
            return
        condition = _lanes_past_ranges_true(state, condition, fake)
    state.device_function.has_device_assert = True
    state.add_statement(
        create(
            ast.Expr,
            value=expr_from_string(
                f"{function}({{condition}}, {{message}})",
                condition=condition,
                message=create(ast.Constant, value=message),
            ),
        )
    )


def _lanes_past_ranges_true(
    state: CodegenState, condition: ast.AST, fake: torch.Tensor
) -> ast.AST:
    """The tensor ``condition``, nonzero if not bool, with the lanes past the
    length of a padded ``hl.arange`` read as true.  The lanes past the end of a
    partial tile read as true already (the op's ``_mask_to``); these are no
    tile's: ``hl.arange(100)`` runs over 128 lanes, and ``hl.arange(100) < 100``
    fails on the last 28.  Triton rejects a condition that is not bool."""
    env = CompileEnvironment.current()
    if fake.dtype != torch.bool:
        condition = expr_from_string("{c} != 0", c=condition)
    assert state.fx_node is not None
    lengths = list(padded_range_lengths(state.fx_node.args[0], fake.ndim))
    for dim, size in enumerate(fake.size()):
        # A dim of a constant size that is no power of two runs over the next
        # one, whatever made it: hl.arange(100).unsqueeze(0) is [1, 100] over
        # [1, 128] lanes, through ops the padded-iota walk does not follow.
        # check_device_asserts rejects one next to a tile's dims, so these
        # logical dims are Triton's (flatten_loops merges only tile dims).
        if lengths[dim] is None and _padded_constant(size):
            lengths[dim] = size
    for dim, length in enumerate(lengths):
        block_id = env.get_block_id(fake.size(dim))
        # A dim of a block of the range's own length (a load at
        # [tile, hl.arange(100)]) is masked by ``_mask_to`` already, as in
        # ReductionLowering._mask_padded_range_lanes.
        if length is None or (
            block_id is not None and env.block_sizes[block_id].size == length
        ):
            continue
        masked = env.backend.mask_padded_range_lanes(
            "{c}", length, dim, fake.ndim, "True"
        )
        if masked is not None:
            condition = expr_from_string(masked, c=condition)
    return condition
