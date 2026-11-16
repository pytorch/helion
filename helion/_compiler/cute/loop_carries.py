"""Loop-carried values of a CuTe device loop.

``prepare_device_loop`` runs before the body of a device loop is rendered
(``CuteBackend.prepare_device_loop``): the lane-merged carry check and the
integer-reduction hoist.  ``finish_device_loop_body`` runs after it
(``CuteBackend.finish_device_loop_body``): the hoist's finalization and the
``_cute_join_cast`` of every carried value whose DSL type may differ from
its FX dtype (``carried_type_joins``).
"""

from __future__ import annotations

import ast
import operator
from typing import TYPE_CHECKING
from typing import cast

import torch

from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from .lane_merged_carries import check_lane_merged_carries

if TYPE_CHECKING:
    from ..device_ir import ForLoopGraphInfo
    from ..inductor_lowering import CodegenState
    from ..tile_strategy import DeviceLoopState


def prepare_device_loop(
    graph_info: ForLoopGraphInfo, state: CodegenState, device_loop: DeviceLoopState
) -> None:
    """Check the lane-merged carries and plan the integer-reduction hoist."""
    from .integer_loop_reduction import prepare

    check_lane_merged_carries(state, graph_info)
    device_loop.integer_reduction_hoist = prepare(graph_info, state, device_loop)


def carried_type_joins(state: CodegenState, initials: dict[int, str]) -> dict[int, str]:
    """The initial variable of each carried value CuTe DSL may retype.

    CuTe DSL requires a loop-carried variable to keep the type it had
    before the loop, but a value's DSL type can differ from its FX dtype:
    a 16-bit value combined with a Python float becomes Float32, and int32
    index math on Int64 shape arguments becomes Int64.  The end of the
    body casts such a value to the type of an alias of its initial value,
    which only tracing knows (``_cute_join_cast``, as for a value one
    branch of a dynamic if reassigns).  fp32/fp64 values keep their DSL
    type, so their loops (accumulators) stay as emitted.  Returns the
    initial variable per loop output.
    """
    if state.fx_node is None:
        return {}
    joins: dict[int, str] = {}
    for user in state.fx_node.users:
        if user.target is not operator.getitem:
            continue
        val = user.meta.get("val")
        if not isinstance(val, torch.Tensor) or val.dtype in (
            torch.float32,
            torch.float64,
        ):
            continue
        index = cast("int", user.args[1])
        if index in initials:
            joins[index] = initials[index]
    return joins


def finish_device_loop_body(
    state: CodegenState,
    device_loop: DeviceLoopState,
    outputs: list[object],
    initials: dict[int, str],
) -> None:
    """Finalize the hoisted reduction and cast the retyped carries in ``outputs``."""
    joins = carried_type_joins(state, initials)
    hoist = device_loop.integer_reduction_hoist
    if hoist is not None and hoist.final_call is not None:
        # The CTA collective after the loop finalizes the carry
        # from the per-thread partial the body carries, which
        # keeps its type.
        hoist.finish(outputs)
        joins = {}
    device_function = state.device_function
    for index, initial in joins.items():
        value = cast("ast.AST", outputs[index])
        if isinstance(
            value, ast.Name
        ) and device_function.cute_state.get_tcgen05_store_value(
            device_function.variable_aliases(value.id)
        ):
            # A tcgen05 accumulator: the MMA lowering owns its type.
            continue
        # The loop statement follows the enclosing statements
        # once its body is complete.
        before = device_function.new_var(f"{initial}_before_loop")
        state.codegen.statements_stack[-2].append(
            statement_from_string(f"{before} = {initial}")
        )
        joined = device_function.new_var(f"{before}_join")
        state.codegen.add_statement(
            statement_from_string(
                f"{joined} = _cute_join_cast({{value}}, {before})",
                value=value,
            )
        )
        outputs[index] = expr_from_string(joined)
