from __future__ import annotations

import ast
import inspect
import itertools
import operator
from typing import TYPE_CHECKING
from typing import Callable
from typing import cast
from typing import overload

import torch
from torch.fx.experimental import proxy_tensor

from .. import exc
from . import _decorators

if TYPE_CHECKING:
    from .._compiler.helper_function import CombineFunction
    from .._compiler.helper_function import CombineFunctionBasic
    from .._compiler.helper_function import CombineFunctionTuple
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

    This function is only needed for user-defined combine functions.
    Standard PyTorch reductions (such as sum, mean, amax, etc.) work
    directly in Helion without requiring this function.

    Args:
        combine_fn: A binary function that combines two elements element-wise.
                   Must be associative and commutative for correct results.
                   Can be tensor->tensor or tuple->tuple function.
        input_tensor: Input tensor or tuple of tensors to reduce
        dim: The dimension along which to reduce (None for all dimensions)
        other: Value for masked/padded elements (default: 0)
               For tuple inputs, can be tuple of values with same length
        keep_dims: If True, reduced dimensions are retained with size 1

    Returns:
        torch.Tensor or tuple[torch.Tensor, ...]: Tensor(s) with reduced dimensions

    See Also:
        - :func:`~helion.language.associative_scan`: For prefix operations

    Note:
        - combine_fn must be associative and commutative
        - For standard reductions, use PyTorch functions directly (faster)
        - Masked elements use the 'other' value during reduction
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


@_decorators.ref(reduce)
def _(
    combine_fn: CombineFunction,
    input_tensor: torch.Tensor | tuple[torch.Tensor, ...],
    dim: int | None = None,
    other: float | tuple[float, ...] = 0,
    keep_dims: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Reference implementation of reduce."""
    from .._compiler.helper_function import extract_helper_function

    # Extract the raw function if it's wrapped in a @helion.kernel decorator
    combine_fn = extract_helper_function(combine_fn)

    is_tuple = isinstance(input_tensor, tuple)

    # Normalize inputs to always work with lists
    if not is_tuple:
        assert isinstance(other, (int, float)), (
            "other must be a scalar for single tensor input"
        )
        input_data = [input_tensor]
        other = (other,)
        # Wrap single-tensor combine function to work with tuples
        original_fn = cast("CombineFunctionBasic", combine_fn)

        def wrapped_combine_fn(
            left_tuple: tuple[torch.Tensor, ...], right_tuple: tuple[torch.Tensor, ...]
        ) -> tuple[torch.Tensor, ...]:
            result = original_fn(left_tuple[0], right_tuple[0])
            return cast("tuple[torch.Tensor, ...]", (result,))

        combine_fn = wrapped_combine_fn
    else:
        input_data = list(input_tensor)
        # Ensure other is a tuple with same length
        if not isinstance(other, tuple):
            other = (other,) * len(input_data)
        else:
            assert len(other) == len(input_data), (
                "other tuple must match input tensor tuple length"
            )

    # Get metadata from first tensor
    first_tensor = input_data[0]
    shape, ndim = first_tensor.shape, first_tensor.ndim

    # Check if unpacked arguments expected (tuple case only)
    if is_tuple:
        sig = inspect.signature(combine_fn)
        num_params = len(sig.parameters)
        expected_unpacked = 2 * len(input_data)  # All elements unpacked

        if num_params == expected_unpacked:
            # Wrap unpacked function to accept packed arguments
            original_fn = cast("CombineFunctionTuple", combine_fn)

            def wrapped_combine_fn2(
                left_tuple: tuple[torch.Tensor, ...],
                right_tuple: tuple[torch.Tensor, ...],
            ) -> tuple[torch.Tensor, ...]:
                return original_fn(*left_tuple, *right_tuple)

            combine_fn = wrapped_combine_fn2

    # Prepare reduction parameters
    if dim is None:
        dims_to_reduce = list(range(ndim))
    else:
        if dim < 0:
            dim = ndim + dim
        dims_to_reduce = [dim]

    # Calculate output shape
    output_shape = []
    for i, s in enumerate(shape):
        if i in dims_to_reduce:
            output_shape.append(1 if keep_dims else None)
        else:
            output_shape.append(s)
    output_shape = [s for s in output_shape if s is not None]

    # Create output tensors (always as list)
    outputs = [
        torch.full(output_shape, other[i], dtype=t.dtype, device=t.device)
        for i, t in enumerate(input_data)
    ]

    # Perform reduction
    # Create index iterators for non-reduced dimensions
    index_iterators = [
        [slice(None)] if i in dims_to_reduce else list(range(shape[i]))
        for i in range(len(shape))
    ]

    # Iterate over all combinations of non-reduced dimensions
    for idx in itertools.product(*index_iterators):
        # Gather values along reduction dimensions
        values_list = []

        # Get ranges for each dimension being reduced
        reduction_ranges = [range(shape[d]) for d in dims_to_reduce]

        # Iterate over all combinations of indices in reduction dimensions
        for reduction_indices in itertools.product(*reduction_ranges):
            full_idx = list(idx)
            # Fill in the reduction dimension indices
            for d, pos in zip(dims_to_reduce, reduction_indices, strict=False):
                full_idx[d] = pos
            values_list.append(tuple(t[tuple(full_idx)] for t in input_data))

        if not values_list:
            continue  # No values to reduce

        # Reduce values
        result = values_list[0]
        tuple_combine_fn = cast(
            "Callable[[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]]",
            combine_fn,
        )
        for values in values_list[1:]:
            result = tuple_combine_fn(result, values)

        # Build output index
        output_idx = tuple(
            0 if isinstance(idx_val, slice) and keep_dims else idx_val
            for idx_val in idx
            if not isinstance(idx_val, slice) or keep_dims
        )

        # Store results
        for i, out in enumerate(outputs):
            out[output_idx] = result[i]

    # Convert back to single tensor if needed
    if not is_tuple:
        return outputs[0]
    return tuple(outputs)


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
    from .._compiler.helper_function import extract_helper_function_name

    is_tuple_input = isinstance(input_tensor, (tuple, list))
    if is_tuple_input:
        assert all(isinstance(t, torch.Tensor) for t in input_tensor), (
            "reduce input must be a tuple of tensors"
        )
    else:
        assert isinstance(input_tensor, torch.Tensor), "reduce input must be a tensor"

    assert callable(combine_fn), "combine_fn must be callable"
    # Extract the function name before wrapping
    original_function_name = extract_helper_function_name(combine_fn)
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
        original_function_name=original_function_name,
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

        mask_node_inputs(actual_node, other=other)  # pyright: ignore[reportArgumentType]

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
    """Register the helper function and return its final name."""
    from .._compiler.device_ir import HelperFunctionGraphInfo
    from .._compiler.host_function import HostFunction

    helper_graph_info = HostFunction.current().device_ir.graphs[combine_graph_id]
    assert isinstance(helper_graph_info, HelperFunctionGraphInfo)
    state.codegen.device_function.register_helper_function(helper_graph_info)
    # Get the final name from the helper manager (which uses the namespace)
    return state.codegen.device_function.helper_manager.get_final_name(
        helper_graph_info
    )


def _create_reduce_expression(
    input_tensor: ast.AST, dim: object, helper_func_name: str, keep_dims: bool
) -> ast.AST:
    """Create the tl.reduce expression."""
    from .._compiler.ast_extension import expr_from_string

    if dim is None:
        # Reduce all dimensions
        if keep_dims:
            template = (
                f"tl.reduce({{input_tensor}}, None, {helper_func_name}, keep_dims=True)"
            )
        else:
            template = f"tl.reduce({{input_tensor}}, None, {helper_func_name})"
        return expr_from_string(
            template,
            input_tensor=input_tensor,
        )
    # Reduce specific dimension
    if keep_dims:
        template = f"tl.reduce({{input_tensor}}, {{dim_value}}, {helper_func_name}, keep_dims=True)"
    else:
        template = f"tl.reduce({{input_tensor}}, {{dim_value}}, {helper_func_name})"
    return expr_from_string(
        template,
        input_tensor=input_tensor,
        dim_value=ast.Constant(value=dim),  # pyright: ignore[reportArgumentType]
    )


def _create_tuple_result_expressions(
    state: CodegenState, reduce_expr: ast.AST
) -> list[ast.AST]:
    """Create getitem expressions for tuple results."""
    from .._compiler.ast_extension import expr_from_string

    raw_input = state.ast_args[1]
    num_elements = len(raw_input) if isinstance(raw_input, tuple) else 2

    return [
        expr_from_string(
            "{reduce_result}[{index}]",
            reduce_result=reduce_expr,
            index=ast.Constant(value=i),
        )
        for i in range(num_elements)
    ]
