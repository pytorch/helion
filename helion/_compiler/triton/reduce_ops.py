"""Triton-backend codegen for ops defined in ``helion.language.reduce_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "triton")``
registrations; ``reduce_ops`` imports it at the bottom so registration keeps the
same eager timing as before.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import cast

from torch._inductor.codegen.simd import constant_repr

from ...language import _decorators
from ...language.reduce_ops import _reduce
from ..ast_extension import create
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ..inductor_lowering import custom_reduction_sources
from ..inductor_lowering import reduction_block_id
from ..node_masking import cached_masked_value
from ..tile_strategy import DeviceLoopState

if TYPE_CHECKING:
    import torch

    from ..inductor_lowering import CodegenState
    from ..reduction_strategy import LoopedReductionStrategy


@_decorators.codegen(_reduce, "triton")
def _(state: CodegenState) -> ast.AST | list[ast.AST]:
    """Generate a persistent or rolled custom reduction."""
    # reduction_strategy imports DeviceIR before helion.Config is initialized.
    from ..device_ir import HelperFunctionGraphInfo
    from ..reduction_strategy import LoopedReductionStrategy

    raw_input = state.ast_args[1]
    input_tensor = (
        create(ast.Tuple, elts=list(raw_input), ctx=ast.Load())
        if isinstance(raw_input, tuple)
        else state.ast_arg(1)
    )
    helper_graph = state.get_graph(cast("int", state.proxy_arg(0)))
    assert isinstance(helper_graph, HelperFunctionGraphInfo)
    combine_func = state.device_function.register_helper_function(helper_graph)
    assert state.fx_node is not None
    block_id = reduction_block_id(state.fx_node)
    loops = (
        state.codegen.active_device_loops.get(block_id, [])
        if block_id is not None
        else []
    )
    loop = loops[-1] if loops else None
    if isinstance(loop, DeviceLoopState) and isinstance(
        loop.strategy, LoopedReductionStrategy
    ):
        result = _codegen_rolled_reduce(state, loop, input_tensor, combine_func)
        state.fx_node.meta["delayed_result"] = True
    else:
        dim = cast("int | None", state.proxy_arg(2))
        keep_dims = state.proxy_arg(3)
        keep_dims_arg = ", keep_dims=True" if keep_dims else ""
        result = expr_from_string(
            f"tl.reduce({{input_tensor}}, {{dim}}, {combine_func}{keep_dims_arg})",
            input_tensor=input_tensor,
            dim=ast.Constant(value=dim),
        )
    if state.proxy_arg(4):
        inputs = cast("tuple[torch.Tensor, ...]", state.proxy_arg(1))
        return [
            expr_from_string("{result}[{index}]", result=result, index=ast.Constant(i))
            for i in range(len(inputs))
        ]
    return result


def _codegen_rolled_reduce(
    state: CodegenState,
    loop: DeviceLoopState,
    input_tensor: ast.AST,
    combine_func: str,
) -> ast.AST:
    """Carry chunk-sized payloads through the loop and reduce after it finishes."""
    assert state.fx_node is not None
    sources = custom_reduction_sources(state.fx_node)
    dim = cast("int", state.proxy_arg(2))
    keep_dims = state.proxy_arg(3)
    is_tuple_input = bool(state.proxy_arg(4))
    fake_inputs = tuple(source.meta["val"] for source in sources)
    device_function = state.device_function
    env = CompileEnvironment.current()
    strategy = cast("LoopedReductionStrategy", loop.strategy)
    accumulators = []
    padding_values = []
    for source, value in zip(sources, fake_inputs, strict=True):
        fill = cached_masked_value(source)
        assert fill is not None
        name = device_function.new_var("custom_reduce_acc", dce=True)
        shape = device_function.tile_strategy.shape_dims(list(value.size()))
        initial = env.backend.full_expr(shape, "0", value.dtype)
        loop.outer_prefix.append(statement_from_string(f"{name} = {initial}"))
        accumulators.append(name)
        padding_values.append(
            env.backend.full_expr(shape, constant_repr(fill), value.dtype)
        )
    targets = ", ".join(accumulators)
    assignment_target = f"({targets},)" if is_tuple_input else targets
    values = cast("ast.Tuple", input_tensor).elts if is_tuple_input else [input_tensor]
    source = ", ".join(ast.unparse(value) for value in values)
    offset = strategy.offset_var(strategy.block_index)
    # The first chunk seeds the state; zero only establishes its tensor type.
    state.add_statement(
        statement_from_string(
            f"if {offset} == 0:\n    {assignment_target} = {ast.unparse(input_tensor)}\n"
            f"else:\n    {assignment_target} = {combine_func}({targets}, {source})"
        )
    )
    padded, dynamic = _emit_reduction_padding(
        state, loop, combine_func, targets, assignment_target, padding_values
    )
    reduction_input = expr_from_string(assignment_target)
    final_combine = combine_func
    if dynamic:
        chunk_var = strategy.block_size_var(strategy.block_index)
        assert chunk_var is not None
        valid = strategy.broadcast_str(
            f"(tl.arange(0, {chunk_var}) < {padded})",
            fake_inputs[0],
            dim % fake_inputs[0].ndim,
        )
        reduction_input = expr_from_string(f"({targets}, {valid})")
        final_combine = _register_valid_combine(
            state, combine_func, len(fake_inputs), is_tuple_input
        )
    keep_dims_arg = ", keep_dims=True" if keep_dims else ""
    reduction = expr_from_string(
        f"tl.reduce({{input_tensor}}, {{dim}}, {final_combine}{keep_dims_arg})",
        input_tensor=reduction_input,
        dim=ast.Constant(value=dim),
    )
    if dynamic and not is_tuple_input:
        reduction = expr_from_string("{result}[0]", result=reduction)
    result = device_function.new_var("custom_reduce_result", dce=True)
    loop.outer_suffix.append(
        statement_from_string(f"{result} = {{reduction}}", reduction=reduction)
    )
    return expr_from_string(result)


def _emit_reduction_padding(
    state: CodegenState,
    loop: DeviceLoopState,
    combine_func: str,
    targets: str,
    assignment_target: str,
    padding_values: list[str],
) -> tuple[str, bool]:
    """Preserve the persistent reduction's number of padded contributions."""
    device_function = state.device_function
    strategy = cast("LoopedReductionStrategy", loop.strategy)
    env = CompileEnvironment.current()
    numel = env.specialize_expr(env.block_sizes[strategy.block_index].numel)
    numel_expr = state.sympy_expr(numel)
    chunk_var = strategy.block_size_var(strategy.block_index)
    assert chunk_var is not None
    dynamic = not numel.is_number
    if not dynamic:
        padded_extent = env.backend.static_rdim_size(int(numel))
        padded = str(padded_extent)
        start = -(-int(numel) // strategy.block_size) * strategy.block_size
        if start >= padded_extent:
            return padded, dynamic
    else:
        padded = device_function.new_var("custom_reduce_padded")
        # Widen before rounding the runtime extent up to a power of two.
        loop.outer_suffix.append(
            statement_from_string(
                f"{padded} = tl.cast(tl.maximum({numel_expr}, 1) - 1, tl.uint64)"
            )
        )
        for shift in (1, 2, 4, 8, 16, 32):
            loop.outer_suffix.append(
                statement_from_string(f"{padded} = {padded} | ({padded} >> {shift})")
            )
        loop.outer_suffix.append(statement_from_string(f"{padded} = {padded} + 1"))
    padding = device_function.new_var("custom_reduce_padding")
    padding_args = ", ".join(padding_values)
    loop.outer_suffix.append(
        statement_from_string(
            f"for {padding} in range((({numel_expr} + {chunk_var} - 1) // {chunk_var}) * {chunk_var}, {padded}, {chunk_var}):\n"
            f"    if {padding} == 0:\n"
            f"        {assignment_target} = ({padding_args})\n"
            f"    else:\n"
            f"        {assignment_target} = {combine_func}({targets}, {padding_args})"
        )
    )
    return padded, dynamic


def _register_valid_combine(
    state: CodegenState,
    combine_func: str,
    input_count: int,
    is_tuple_input: bool,
) -> str:
    """Extend the combine function with validity flags without assuming an identity."""
    left = [f"left_{i}" for i in range(input_count)]
    right = [f"right_{i}" for i in range(input_count)]
    args = ", ".join([*left, "left_valid", *right, "right_valid"])
    call_args = ", ".join([*left, *right])
    outputs = []
    for index, (left_value, right_value) in enumerate(zip(left, right, strict=True)):
        combined = f"combined[{index}]" if is_tuple_input else "combined"
        outputs.append(
            f"tl.where(left_valid, tl.where(right_valid, {combined}, {left_value}), {right_value})"
        )
    name = state.device_function.new_var("custom_reduce_valid_combine")
    state.codegen.module_statements.append(
        cast(
            "ast.stmt",
            statement_from_string(
                f"@triton.jit\ndef {name}({args}):\n"
                f"    combined = {combine_func}({call_args})\n"
                f"    return {', '.join([*outputs, 'left_valid | right_valid'])}"
            ),
        )
    )
    return name
