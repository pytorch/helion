from __future__ import annotations

import ast
import operator
from typing import TYPE_CHECKING
from typing import cast
from typing import overload

import torch
from torch.fx.experimental import proxy_tensor

from .. import exc
from . import _decorators

if TYPE_CHECKING:
    from .._compiler.helper_function import CombineFunction
    from .._compiler.inductor_lowering import CodegenState


__all__ = ["reduce"]


@overload
@_decorators.api(is_device_only=True)
def reduce(
    combine_fn: CombineFunction,
    input_tensor: torch.Tensor,
    dim: int | None = None,
    other: float = 0,
    keep_dims: bool = False,
) -> torch.Tensor: ...


@overload
@_decorators.api(is_device_only=True)
def reduce(
    combine_fn: CombineFunction,
    input_tensor: tuple[torch.Tensor, ...],
    dim: int | None = None,
    other: float | tuple[float, ...] = 0,
    keep_dims: bool = False,
) -> tuple[torch.Tensor, ...]: ...


@_decorators.api(is_device_only=True)
def reduce(
    combine_fn: CombineFunction,
    input_tensor: torch.Tensor | tuple[torch.Tensor, ...],
    dim: int | None = None,
    other: float | tuple[float, ...] = 0,
    keep_dims: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """
    Applies a reduction operation along a specified dimension or all dimensions.

    Args:
        combine_fn: A binary function that combines two elements element-wise.
                   Can be tensor->tensor or tuple->tuple function.
        input_tensor: Input tensor or tuple of tensors to reduce.
        dim: The dimension along which the reduction should be done.
             If None, reduce all dimensions.
        other: Value to use for masked/padded elements. For tuple inputs,
               can be a tuple of values with same length as input tuple.
        keep_dims: If True, the reduced dimensions are retained with size 1.

    Returns:
        A tensor or tuple of tensors with reduced dimensions.
    """
    raise exc.NotInsideKernel


@_decorators.register_fake(reduce)
def _(
    combine_fn: CombineFunction,
    input_tensor: torch.Tensor | tuple[torch.Tensor, ...],
    dim: int | None = None,
    other: float | tuple[float, ...] = 0,
    keep_dims: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Fake implementation that returns fake tensors with reduced shape."""
    if isinstance(input_tensor, (tuple, list)):
        return tuple(_fake_reduce_tensor(t, dim, keep_dims) for t in input_tensor)
    return _fake_reduce_tensor(input_tensor, dim, keep_dims)


def _fake_reduce_tensor(
    tensor: torch.Tensor, dim: int | None, keep_dims: bool
) -> torch.Tensor:
    """Helper to create a fake tensor with reduced dimensions."""
    if dim is None:
        # Reduce all dimensions
        if keep_dims:
            return torch.empty(
                [1] * tensor.ndim, dtype=tensor.dtype, device=tensor.device
            )
        return torch.empty([], dtype=tensor.dtype, device=tensor.device)
    # Reduce specific dimension
    new_shape = [*tensor.shape]
    # Handle negative dimension indexing
    if dim < 0:
        dim = tensor.ndim + dim

    if keep_dims:
        new_shape[dim] = 1
    else:
        new_shape.pop(dim)
    return torch.empty(new_shape, dtype=tensor.dtype, device=tensor.device)


@_decorators.register_to_device_ir(reduce)
def _(
    tracer: proxy_tensor.PythonKeyTracer,
    combine_fn: CombineFunction,
    input_tensor: torch.Tensor | tuple[torch.Tensor, ...],
    dim: int | None = None,
    other: float | tuple[float, ...] = 0,
    keep_dims: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """
    Device IR implementation that handles tracing for reduce. We map
    reduce to _reduce, with a pre-traced graph for the combine function.
    """
    from .._compiler.device_ir import DeviceIR
    from .._compiler.device_ir import HelperFunctionGraphInfo
    from .._compiler.device_ir import args_to_proxies
    from .._compiler.device_ir import select_decomp_table
    from .._compiler.helper_function import create_combine_function_wrapper

    is_tuple_input = isinstance(input_tensor, (tuple, list))
    if is_tuple_input:
        assert all(isinstance(t, torch.Tensor) for t in input_tensor), (
            "reduce input must be a tuple of tensors"
        )
    else:
        assert isinstance(input_tensor, torch.Tensor), "reduce input must be a tensor"

    assert callable(combine_fn), "combine_fn must be callable"
    combine_fn = create_combine_function_wrapper(
        combine_fn, is_tuple_input=is_tuple_input, target_format="tuple"
    )

    # Create fake inputs for the combine function
    if is_tuple_input:
        # For tuple inputs, create two tuples of fake tensors for left and right args
        left_fake_tensors = []
        right_fake_tensors = []
        for tensor in input_tensor:
            left_fake_tensors.append(
                torch.empty([1], dtype=tensor.dtype, device=tensor.device)
            )
            right_fake_tensors.append(
                torch.empty([1], dtype=tensor.dtype, device=tensor.device)
            )
        # The combine function expects (left_tuple, right_tuple)
        fake_inputs = [tuple(left_fake_tensors), tuple(right_fake_tensors)]
    else:
        # For single tensor inputs, create two different fake tensors for left and right args
        left_fake_tensor = torch.empty(
            [1], dtype=input_tensor.dtype, device=input_tensor.device
        )
        right_fake_tensor = torch.empty(
            [1], dtype=input_tensor.dtype, device=input_tensor.device
        )
        fake_inputs = [left_fake_tensor, right_fake_tensor]

    combine_graph = proxy_tensor.make_fx(
        combine_fn, decomposition_table=select_decomp_table()
    )(*fake_inputs).graph
    combine_graph_id = DeviceIR.current().add_graph(
        combine_graph,
        HelperFunctionGraphInfo,
        node_args=[],
    )

    # Validate other parameter for mask_node_inputs
    if is_tuple_input:
        assert isinstance(input_tensor, (tuple, list))

        # Handle other parameter for tuple inputs
        if isinstance(other, (tuple, list)):
            if len(other) != len(input_tensor):
                raise ValueError(
                    f"other tuple length {len(other)} must match input tensor length {len(input_tensor)}"
                )
            # For tuple inputs with tuple others, mask_node_inputs doesn't directly support this
            # We'll handle this in a different way below
        else:
            # Broadcast single other value to all tensors - mask_node_inputs will handle this
            pass
    else:
        # Single tensor case
        if isinstance(other, (tuple, list)):
            raise ValueError("other must be a scalar for single tensor input")

    # Create the reduce tracing operation without other values (masking will be handled by mask_node_inputs)
    reduce_args = (
        combine_graph_id,
        input_tensor,
        dim,
        keep_dims,
        is_tuple_input,
    )
    proxy_args, proxy_kwargs = args_to_proxies(tracer, reduce_args)
    proxy_out = tracer.create_proxy(
        "call_function",
        _reduce,
        proxy_args,
        proxy_kwargs,
    )

    # Apply masking to the input tensors in the proxy node
    from .._compiler.node_masking import apply_masking

    # Get the actual node from the proxy and apply masking
    actual_node = proxy_out.node

    if is_tuple_input and isinstance(other, (tuple, list)):
        # For tuple inputs with tuple others, apply masking to each tensor separately
        input_arg = actual_node.args[1]
        assert isinstance(input_arg, (tuple, list))
        masked_tensors = []
        for tensor_node, other_val in zip(input_arg, other, strict=True):
            assert isinstance(tensor_node, torch.fx.Node)
            masked_tensor = apply_masking(
                tensor_node, base_node=actual_node, other=other_val
            )
            masked_tensors.append(masked_tensor)
        # Update the args with masked tensors
        actual_node.args = (
            actual_node.args[0],
            tuple(masked_tensors),
            *actual_node.args[2:],
        )
    else:
        # For single tensor or single other value, use mask_node_inputs
        from .._compiler.node_masking import mask_node_inputs

        mask_node_inputs(actual_node, other=other)

    # Create output tensors with reduced shape
    if is_tuple_input:
        output_tensors = []
        assert isinstance(input_tensor, (tuple, list))
        for i, tensor in enumerate(input_tensor):
            reduced_tensor = _fake_reduce_tensor(tensor, dim, keep_dims)
            element_proxy = tracer.create_proxy(
                "call_function",
                operator.getitem,
                (proxy_out, i),
                {},
            )
            proxy_tensor.track_tensor_tree(
                reduced_tensor, element_proxy, constant=None, tracer=tracer
            )
            output_tensors.append(reduced_tensor)
        return tuple(output_tensors)

    output_tensor = _fake_reduce_tensor(input_tensor, dim, keep_dims)
    proxy_tensor.track_tensor_tree(
        output_tensor, proxy_out, constant=None, tracer=tracer
    )
    return output_tensor


@_decorators.api()
def _reduce(
    combine_graph_id: int,
    input_tensor: torch.Tensor | tuple[torch.Tensor, ...],
    dim: int | None = None,
    keep_dims: bool = False,
    is_tuple_input: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Device IR implementation of reduce, not meant to be called directly."""
    raise AssertionError("this should never be called")


@_decorators.register_fake(_reduce)
def _(
    combine_graph_id: int,
    input_tensor: torch.Tensor | tuple[torch.Tensor, ...],
    dim: int | None = None,
    keep_dims: bool = False,
    is_tuple_input: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Fake implementation that returns tensors with reduced shape."""
    if is_tuple_input:
        assert isinstance(input_tensor, (tuple, list)), input_tensor
        return tuple(_fake_reduce_tensor(t, dim, keep_dims) for t in input_tensor)
    assert isinstance(input_tensor, torch.Tensor), input_tensor
    return _fake_reduce_tensor(input_tensor, dim, keep_dims)


@_decorators.codegen(_reduce)
def _(state: CodegenState) -> ast.AST | list[ast.AST]:
    """Generate code for reduce with combine function."""

    combine_graph_id = state.proxy_arg(0)
    dim = state.proxy_arg(2)
    keep_dims = state.proxy_arg(3)
    is_tuple_input = state.proxy_arg(4)

    # Input tensor is already masked, so we can use it directly
    if is_tuple_input:
        # For tuple inputs, we need to handle the tuple structure
        input_tensor = state.ast_args[1]
        if isinstance(input_tensor, tuple):
            from .._compiler.ast_extension import create

            input_tensor = create(ast.Tuple, elts=list(input_tensor), ctx=ast.Load())
        else:
            input_tensor = state.ast_arg(1)
    else:
        input_tensor = state.ast_arg(1)
    helper_func_name = _register_helper_function(state, cast("int", combine_graph_id))
    reduce_expr = _create_reduce_expression(
        input_tensor, dim, helper_func_name, bool(keep_dims)
    )

    if is_tuple_input:
        return _create_tuple_result_expressions(state, reduce_expr)
    return reduce_expr


def _register_helper_function(state: CodegenState, combine_graph_id: int) -> str:
    """Register the helper function and return its name."""
    from .._compiler.host_function import HostFunction

    helper_graph_info = HostFunction.current().device_ir.graphs[combine_graph_id]
    state.codegen.device_function.register_helper_function(helper_graph_info)
    return helper_graph_info.name


def _create_reduce_expression(
    input_tensor: ast.AST, dim: object, helper_func_name: str, keep_dims: bool
) -> ast.AST:
    """Create the tl.reduce expression."""
    from .._compiler.ast_extension import expr_from_string

    if dim is None:
        # Reduce all dimensions
        if keep_dims:
            template = (
                f"tl.reduce(input_tensor, None, {helper_func_name}, keep_dims=True)"
            )
        else:
            template = f"tl.reduce(input_tensor, None, {helper_func_name})"
        return expr_from_string(
            template,
            input_tensor=input_tensor,
        )
    # Reduce specific dimension
    if keep_dims:
        template = (
            f"tl.reduce(input_tensor, dim_value, {helper_func_name}, keep_dims=True)"
        )
    else:
        template = f"tl.reduce(input_tensor, dim_value, {helper_func_name})"
    return expr_from_string(
        template,
        input_tensor=input_tensor,
        dim_value=ast.Constant(value=dim),
    )


def _create_tuple_result_expressions(
    state: CodegenState, reduce_expr: ast.AST
) -> list[ast.AST]:
    """Create getitem expressions for tuple results."""
    from .._compiler.ast_extension import expr_from_string

    raw_input = state.ast_args[1]
    num_elements = len(raw_input) if isinstance(raw_input, tuple) else 2

    return [
        expr_from_string(f"reduce_result[{i}]", reduce_result=reduce_expr)
        for i in range(num_elements)
    ]
