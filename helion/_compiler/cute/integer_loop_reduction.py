"""Exact integer min/max recurrences with one collective after a tiled loop.

The logical proof limits all intermediate reduction users to one recurrence.
The physical proof is performed by the ordinary BlockReductionStrategy while
its complete CTA owner is live. Producer operations and loads stay in place.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from itertools import starmap
import operator
from typing import TYPE_CHECKING

import sympy
import torch

from ...language import _tracing_ops
from ...language import creation_ops
from ...language import inline_asm_ops
from ...language import memory_ops
from ...language import tile_ops
from ...language import view_ops
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string

if TYPE_CHECKING:
    from ..device_ir import ForLoopGraphInfo
    from ..device_ir import GraphInfo
    from ..inductor_lowering import CodegenState
    from ..tile_strategy import DeviceLoopState

KEY = "cute_integer_loop_reduction"


@dataclass(frozen=True)
class Recurrence:
    reduction: torch.fx.Node
    operation: str
    dtype: torch.dtype


def recurrence(info: GraphInfo) -> Recurrence | None:
    """One pure tiled-loop recurrence, with no early or carry-dependent use."""
    from ..device_ir import ForLoopGraphInfo

    if type(info) is not ForLoopGraphInfo or len(info.block_ids) != 1:
        return None
    placeholders = list(info.graph.find_nodes(op="placeholder"))
    outputs = list(info.graph.find_nodes(op="output"))
    if len(placeholders) != 1 or len(info.node_args) != 1 or len(outputs) != 1:
        return None
    values = outputs[0].args[0]
    if not isinstance(values, (tuple, list)) or len(values) != 1:
        return None
    combine = values[0]
    operations = {
        torch.ops.aten.maximum.default: (torch.ops.aten.amax.default, "max"),
        torch.ops.aten.minimum.default: (torch.ops.aten.amin.default, "min"),
    }
    if not isinstance(combine, torch.fx.Node) or combine.target not in operations:
        return None
    reduce_target, operation = operations[combine.target]
    if len(combine.args) != 2 or set(combine.users) != {outputs[0]}:
        return None
    carry, reduction = combine.args
    if isinstance(carry, torch.fx.Node) and carry.target is reduce_target:
        carry, reduction = reduction, carry
    if not (
        isinstance(carry, torch.fx.Node)
        and carry.target is _tracing_ops._new_var
        and carry.args == (placeholders[0],)
        and set(carry.users) == {combine}
        and isinstance(reduction, torch.fx.Node)
        and reduction.target is reduce_target
        and set(reduction.users) == {combine}
    ):
        return None
    value = reduction.meta.get("val")
    initial = info.node_args[0]
    if not isinstance(value, torch.Tensor) or value.dtype not in (
        torch.int32,
        torch.int64,
    ):
        return None
    if not all(
        isinstance(n.meta.get("val"), torch.Tensor)
        and n.meta["val"].dtype == value.dtype
        for n in (carry, combine, initial)
    ):
        return None
    identity = (
        torch.iinfo(value.dtype).min
        if operation == "max"
        else torch.iinfo(value.dtype).max
    )
    if initial.target is creation_ops.full:
        fill = initial.args[1]
    elif initial.target is torch.ops.aten.scalar_tensor.default:
        fill = initial.args[0]
    else:
        return None
    if type(fill) is not int or fill != identity:
        return None
    # Shape metadata may refer to the carry; its contents must not feed the
    # producer, bounds, addresses, masks, or another recurrence.
    if any(
        u is not carry and u.target is not torch.ops.aten.sym_size.int
        for u in placeholders[0].users
    ):
        return None
    allowed = {
        _tracing_ops._new_var,
        _tracing_ops._host_tensor,
        _tracing_ops._get_symnode,
        _tracing_ops._mask_to,
        creation_ops.full,
        memory_ops.load,
        tile_ops.tile_index,
        view_ops.subscript,
        torch.ops.aten.sym_size.int,
        torch.ops.aten.scalar_tensor.default,
        torch.ops.aten.view.dtype,
        torch.ops.aten.alias.default,
    }
    for node in info.graph.nodes:
        if node.op in ("placeholder", "output") or node is reduction:
            continue
        if node.op != "call_function":
            return None
        if node.target in allowed:
            continue
        if node.target is inline_asm_ops.inline_asm_elementwise:
            # The ordinary API's explicit purity contract, never PTX matching.
            if len(node.args) > 4 and node.args[4] is True:
                continue
            return None
        if (
            isinstance(node.target, torch._ops.OpOverload)
            and torch.Tag.pointwise in node.target.tags
            and torch.Tag.nondeterministic_seeded not in node.target.tags
            and not node.target._schema.is_mutable
        ):
            continue
        return None
    return Recurrence(reduction, operation, value.dtype)


@dataclass
class Hoist:
    proof: Recurrence
    loop: DeviceLoopState
    extent: int
    final_call: ast.AST | None = None

    def finish(self, output: list[object]) -> None:
        if self.final_call is None:
            return
        # GraphInterpreter always lifts the proved tensor recurrence to a name.
        assert len(output) == 1 and isinstance(output[0], ast.Name)
        assert isinstance(self.final_call, ast.Call)
        self.final_call.args[0] = output[0]
        self.loop.outer_suffix.append(
            statement_from_string(
                "{carry} = {call}", carry=output[0], call=self.final_call
            )
        )


def prepare(
    info: ForLoopGraphInfo, state: CodegenState, loop: DeviceLoopState
) -> Hoist | None:
    from ..device_ir import RootGraphInfo
    from ..tile_strategy import DeviceGridState

    cg = state.codegen
    if not state.config.config.get(KEY, False):
        return None
    proof = recurrence(info)
    if proof is None or state.fx_node is None:
        return None
    # A direct root loop only. No enclosing branch/device loop and no lane
    # wrappers can turn the final CTA collective into divergent execution.
    if not any(
        isinstance(g, RootGraphInfo) and g.graph is state.fx_node.graph
        for g in cg.codegen_graphs
    ):
        return None
    if (
        cg._cute_branch_path
        or cg.cute_synthetic_arange_axes
        or cg.cute_synthetic_arange_lane_exprs
    ):
        return None
    if any(
        not isinstance(x, DeviceGridState) or x.has_lane_loops()
        for loops in cg.active_device_loops.values()
        for x in loops
    ):
        return None
    if loop.lane_loop_blocks or len(loop.block_id_to_info) != 1:
        return None
    dim = loop.block_id_to_info[info.block_ids[0]]
    # Exact static bounds, not tracing hints. Unknown/data-dependent bounds
    # retain the old schedule; later extensions can add a uniformity proof.
    if not isinstance(dim.begin_expr, sympy.Integer) or not isinstance(
        dim.end_expr, sympy.Integer
    ):
        return None
    if dim.end_expr <= dim.begin_expr:
        return None
    # Parent output must be the usual one-slot extraction and matching phi.
    users = list(state.fx_node.users)
    if (
        len(users) != 1
        or users[0].target is not operator.getitem
        or users[0].args[1] != 0
    ):
        return None
    if len(state.fx_node.args) > 4 and state.fx_node.args[4] not in ([None], [1]):
        return None
    captured = state.fx_node.args[3]
    if not isinstance(captured, (list, tuple)) or len(captured) != 1:
        return None
    uses = list(users[0].users)
    if (
        len(uses) != 1
        or uses[0].target is not _tracing_ops._phi
        or uses[0].args != (captured[0], users[0])
    ):
        return None
    return Hoist(proof, loop, int(dim.end_expr) - int(dim.begin_expr))


def defer_collective(
    state: CodegenState,
    *,
    block: int,
    operation: str,
    dtype: torch.dtype,
    identity: str,
    pre: int,
    span: int,
    groups: int,
) -> bool:
    """Capture a fully proved two-stage collective; leave every other path alone."""
    from ..tile_strategy import DeviceLoopState

    cg = state.codegen
    active = cg.active_device_loops.get(block, ())
    if len(active) != 1 or not isinstance(active[0], DeviceLoopState):
        return False
    loop = active[0]
    plan = loop.integer_reduction_hoist
    if plan is None or plan.final_call is not None:
        return False
    if (
        state.fx_node is not plan.proof.reduction
        or dtype != plan.proof.dtype
        or operation != plan.proof.operation
    ):
        return False
    if pre != 1 or groups != 1 or span <= 32 or span > 1024 or span % 32:
        return False
    # Only physical x is active, the tiled extent equals that complete CTA,
    # and neither earlier nor future planned axes add another group.
    if loop.block_thread_axes != {block: 0} or loop.thread_axis_sizes != {0: span}:
        return False
    planned = cg.device_function.tile_strategy.thread_block_dims()
    if tuple(starmap(max, zip(planned, cg.max_thread_block_dims, strict=True))) != (
        span,
        1,
        1,
    ):
        return False
    if cg.cute_synthetic_arange_axes or cg.cute_synthetic_arange_lane_exprs:
        return False
    if plan.extent <= span:
        return False
    # range must actually use one physical tile per step, with no serialized
    # lane/subtile wrapper. The configured block extent is the existing owner.
    from ..tile_strategy import BlockSizeTileStrategy

    if not isinstance(
        loop.strategy, BlockSizeTileStrategy
    ) or loop.strategy.block_size not in (span, [span]):
        return False
    lane = "cutlass.Int32(cute.arch.thread_idx()[0])"
    plan.final_call = expr_from_string(
        f"_cute_grouped_reduce_shared_two_stage(0, {operation!r}, {identity}, {lane}, ({lane}) % {span}, 0, pre=1, group_span={span}, group_count=1)"
    )
    return True
