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
from ..tile_strategy import NDTileStrategy
from .device_range import int32_range_guard
from .device_range import needs_wide_increment
from .lane_merged_carries import check_lane_merged_carries
from .state import cute_codegen_state

if TYPE_CHECKING:
    from ..device_ir import ForLoopGraphInfo
    from ..generate_ast import GenerateAST
    from ..inductor_lowering import CodegenState
    from ..tile_strategy import DeviceLoopState


def _check_device_range(state: CodegenState, loop: DeviceLoopState) -> None:
    """CuTe's absolute ND loop uses range(Int32(begin), Int32(end), Int32(step))."""
    if not isinstance(loop.strategy, NDTileStrategy):
        return
    operands = []
    for arg in (1, 2, 4):  # begin, exclusive end, optional explicit step
        if arg >= len(state.proxy_args) or state.proxy_arg(arg) is None:
            continue
        bounds, expressions = state.proxy_arg(arg), state.ast_args[arg]
        assert isinstance(bounds, (tuple, list)) and isinstance(
            expressions, (tuple, list)
        )
        for bound, expr in zip(bounds, expressions, strict=True):
            if bound is not None:
                value = ast.unparse(expr) if isinstance(expr, ast.AST) else str(expr)
                operands.append((bound, value, arg == 4))
    guard = int32_range_guard(operands, state.device_function)
    begins, ends = state.proxy_arg(1), state.proxy_arg(2)
    explicit = state.proxy_arg(4) if len(state.proxy_args) > 4 else None
    assert isinstance(begins, (list, tuple)) and isinstance(ends, (list, tuple))
    wide: dict[str, int | None] = {}
    for index, block in enumerate(loop.strategy.block_ids):
        step = explicit[index] if isinstance(explicit, (list, tuple)) else None
        if step is None:
            step = state.device_function.resolved_block_size(block)
        if needs_wide_increment(begins[index], ends[index], step):
            wide[loop.strategy.offset_var(block)] = (
                step if isinstance(step, int) else None
            )
    if guard is not None or wide:
        cute_codegen_state(state.codegen).gather_device_range_guards[
            loop.region_instance
        ] = guard, wide


def prepare_gather_range(loop: DeviceLoopState, cg: GenerateAST) -> None:
    ranges = cute_codegen_state(cg).gather_device_range_guards
    if (plan := ranges.pop(loop.region_instance, None)) is None:
        return
    guard, wide = plan
    if guard is not None:
        loop.outer_prefix.append(guard)
    for header in loop.loop_headers():
        if isinstance(header.target, ast.Name) and header.target.id in wide:
            # Keep each live coordinate Int32; only the final SDK increment
            # can exceed its range, including in a reflected descending loop.
            assert isinstance(header.iter, ast.Call)
            begin, end, delta = header.iter.args
            original = header.target.id
            counter = cg.device_function.new_var("gather_counter")
            if (stride := wide[original]) is not None and stride > 0:
                # A literal positive step bypasses CuTe's reflection and does
                # not need a division to calculate the iteration count.
                iteration = expr_from_string(
                    f"range(cutlass.Int64({{begin}}), cutlass.Int64({{end}}), {stride})",
                    begin=begin,
                    end=end,
                )
                coordinate = counter
            else:
                start, stop, step = (
                    cg.device_function.new_var(f"gather_{name}")
                    for name in ("start", "stop", "step")
                )
                for name, arg in zip(
                    (start, stop, step), (begin, end, delta), strict=True
                ):
                    loop.outer_prefix.append(
                        statement_from_string(
                            f"{name} = cutlass.Int64({{arg}})", arg=arg
                        )
                    )
                distance = f"({stop} - {start} if {step} > 0 else {start} - {stop})"
                magnitude = f"max({step}, -{step})"
                iteration = expr_from_string(
                    f"range(cutlass.Int64(0), ({distance} + {magnitude} - 1) // {magnitude}, 1)"
                )
                coordinate = f"{start} + {counter} * {step}"
            assert isinstance(iteration, ast.Call)
            header.iter = iteration
            header.target.id = counter
            offset = statement_from_string(f"{original} = cutlass.Int32({coordinate})")
            header.body.insert(0, offset)
            if header.body is loop.inner_statements:
                # The lane scheduler moves registered setup before other
                # statements. Keep this definition ahead of its indices and mask.
                loop.lane_setup_statements.insert(0, offset)


def prepare_device_loop(
    graph_info: ForLoopGraphInfo, state: CodegenState, device_loop: DeviceLoopState
) -> None:
    """Check the lane-merged carries and plan the integer-reduction hoist."""
    from .integer_loop_reduction import prepare

    _check_device_range(state, device_loop)
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
