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


def _prepare_combine_function(
    combine_fn: CombineFunction,
) -> tuple[CombineFunction, int]:
    """Extract and prepare the combine function, returning (function, param_count)."""
    import inspect

    from ..runtime.kernel import Kernel

    # Extract underlying function if it's a Kernel
    if isinstance(combine_fn, Kernel):
        combine_fn = combine_fn.fn

    # Get parameter count for determining call format
    sig = inspect.signature(combine_fn)
    param_count = len(sig.parameters)

    return combine_fn, param_count


def _call_combine_function(
    combine_fn: CombineFunction,
    param_count: int,
    left: torch.Tensor | tuple,
    right: torch.Tensor | tuple,
) -> torch.Tensor | tuple:
    """Call combine function with proper format based on parameter count."""
    if isinstance(left, tuple) and isinstance(right, tuple) and param_count > 2:
        # Unpacked format - pass tuple elements as separate arguments
        return combine_fn(*left, *right)
    # Regular format - pass arguments directly
    return combine_fn(left, right)  # type: ignore[arg-type]


def _create_empty_result(
    other: float | tuple[float, ...],
    input_tensor: torch.Tensor | tuple[torch.Tensor, ...],
    index: int | None = None,
) -> torch.Tensor:
    """Create an empty result tensor with the appropriate value."""
    if isinstance(input_tensor, tuple):
        assert index is not None
        dtype = input_tensor[index].dtype
        device = input_tensor[index].device
        if isinstance(other, tuple):
            value = other[index]
        else:
            value = other
    else:
        dtype = input_tensor.dtype
        device = input_tensor.device
        value = other

    return torch.tensor(value, dtype=dtype, device=device)


def _transpose_for_reduction(
    tensor: torch.Tensor, dim: int
) -> tuple[torch.Tensor, list[int]]:
    """Transpose tensor to move reduction dimension to end and return permutation."""
    perm = list(range(tensor.ndim))
    perm[dim], perm[-1] = perm[-1], perm[dim]
    transposed = tensor.permute(perm)
    return transposed, perm


def _get_inverse_permutation(perm: list[int], ndim: int) -> list[int]:
    """Get inverse permutation for restoring original dimension order."""
    inv_perm = list(range(ndim))
    for i, p in enumerate(perm):
        inv_perm[p] = i
    return inv_perm


def _reduce_single_row(
    row: torch.Tensor,
    combine_fn: CombineFunction,
    param_count: int,
    other: float,
    input_tensor: torch.Tensor,
) -> torch.Tensor:
    """Reduce a single row using the combine function."""
    if row.numel() == 0:
        return _create_empty_result(other, input_tensor)

    result = row[0]
    for i in range(1, row.numel()):
        result = _call_combine_function(combine_fn, param_count, result, row[i])
    assert isinstance(result, torch.Tensor)
    return result


def _reduce_tuple_row(
    rows: list[torch.Tensor],
    combine_fn: CombineFunction,
    param_count: int,
    other: float | tuple[float, ...],
    input_tensor: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    """Reduce a row of tuple tensors using the combine function."""
    if rows[0].numel() == 0:
        return tuple(
            _create_empty_result(other, input_tensor, i)
            for i in range(len(input_tensor))
        )

    # Initialize with first element
    result = tuple(row[0] for row in rows)
    # Combine with remaining elements
    for i in range(1, rows[0].numel()):
        next_elem = tuple(row[i] for row in rows)
        result = _call_combine_function(combine_fn, param_count, result, next_elem)
    assert isinstance(result, tuple)
    return result  # type: ignore[return-value]


def _reduce_all_dims_single(
    input_tensor: torch.Tensor,
    combine_fn: CombineFunction,
    param_count: int,
    other: float,
    keep_dims: bool,
) -> torch.Tensor:
    """Reduce all dimensions for a single tensor."""
    flat = input_tensor.flatten()
    if flat.numel() == 0:
        return _create_empty_result(other, input_tensor)

    result = flat[0]
    for i in range(1, flat.numel()):
        result = _call_combine_function(combine_fn, param_count, result, flat[i])

    if keep_dims:
        assert isinstance(result, torch.Tensor)
        result = result.reshape([1] * len(input_tensor.shape))
    assert isinstance(result, torch.Tensor)
    return result


def _reduce_all_dims_tuple(
    input_tensor: tuple[torch.Tensor, ...],
    combine_fn: CombineFunction,
    param_count: int,
    other: float | tuple[float, ...],
    keep_dims: bool,
) -> tuple[torch.Tensor, ...]:
    """Reduce all dimensions for tuple of tensors."""
    # Flatten all tensors in the tuple
    flat_tensors = [t.flatten() for t in input_tensor]

    # Start with the first element
    result = tuple(flat[0:1] for flat in flat_tensors)

    # Combine with remaining elements
    for i in range(1, flat_tensors[0].numel()):
        next_elem = tuple(flat[i : i + 1] for flat in flat_tensors)
        result = _call_combine_function(combine_fn, param_count, result, next_elem)

    # Handle output shape
    if not keep_dims:
        result = tuple(r.item() if r.numel() == 1 else r for r in result)
        # Convert scalars back to tensors to maintain consistent return type
        result = tuple(
            torch.tensor(r) if not isinstance(r, torch.Tensor) else r for r in result
        )
    else:
        result = tuple(
            r.reshape([1] * len(t.shape))
            for r, t in zip(result, input_tensor, strict=True)
        )

    assert isinstance(result, tuple)
    return result


def _reduce_specific_dim_single(
    input_tensor: torch.Tensor,
    combine_fn: CombineFunction,
    param_count: int,
    dim: int,
    other: float,
    keep_dims: bool,
) -> torch.Tensor:
    """Reduce a specific dimension for a single tensor."""
    # Transpose to move reduction dimension to end
    transposed, perm = _transpose_for_reduction(input_tensor, dim)

    # Get result shape
    result_shape = list(transposed.shape[:-1])
    if keep_dims:
        result_shape.append(1)

    # Flatten all dimensions except the last one
    flat_shape = (-1, transposed.shape[-1])
    flat = transposed.reshape(flat_shape)

    # Apply reduction along the last dimension
    results = [
        _reduce_single_row(row, combine_fn, param_count, other, input_tensor)
        for row in flat
    ]

    # Reshape back
    result = torch.stack(results).reshape(result_shape)

    # Permute back if keep_dims
    if keep_dims:
        inv_perm = _get_inverse_permutation(perm, input_tensor.ndim)
        result = result.permute(inv_perm)

    return result


def _reduce_specific_dim_tuple(
    input_tensor: tuple[torch.Tensor, ...],
    combine_fn: CombineFunction,
    param_count: int,
    dim: int,
    other: float | tuple[float, ...],
    keep_dims: bool,
) -> tuple[torch.Tensor, ...]:
    """Reduce a specific dimension for tuple of tensors."""
    # Transpose all tensors to move reduction dimension to end
    transposed_data = [_transpose_for_reduction(t, dim) for t in input_tensor]
    transposed_tensors = [data[0] for data in transposed_data]
    perms = [data[1] for data in transposed_data]

    # Get result shape from first tensor
    result_shape = list(transposed_tensors[0].shape[:-1])
    if keep_dims:
        result_shape.append(1)

    # Flatten all dimensions except the last one
    flat_shape = (-1, transposed_tensors[0].shape[-1])
    flat_tensors = [t.reshape(flat_shape) for t in transposed_tensors]

    # Apply reduction along the last dimension for each position
    result_lists = [[] for _ in input_tensor]
    for row_idx in range(flat_tensors[0].shape[0]):
        # Get the row from each tensor
        rows = [flat_tensors[i][row_idx] for i in range(len(input_tensor))]

        # Reduce the row
        row_results = _reduce_tuple_row(
            rows, combine_fn, param_count, other, input_tensor
        )

        # Append results
        for i, r in enumerate(row_results):
            result_lists[i].append(r)

    # Stack and reshape results
    results = []
    for i, result_list in enumerate(result_lists):
        result_tensor = torch.stack(result_list).reshape(result_shape)

        # Permute back if keep_dims
        if keep_dims:
            inv_perm = _get_inverse_permutation(perms[i], input_tensor[i].ndim)
            result_tensor = result_tensor.permute(inv_perm)

        results.append(result_tensor)

    return tuple(results)


@_decorators.ref(reduce)
def _(
    combine_fn: CombineFunction,
    input_tensor: torch.Tensor | tuple[torch.Tensor, ...],
    dim: int | None = None,
    other: float | tuple[float, ...] = 0,
    keep_dims: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Reference implementation of reduce operation."""
    # Prepare combine function and get parameter count
    combine_fn, param_count = _prepare_combine_function(combine_fn)

    # Handle tuple inputs
    if isinstance(input_tensor, tuple):
        if dim is None:
            return _reduce_all_dims_tuple(
                input_tensor, combine_fn, param_count, other, keep_dims
            )
        return _reduce_specific_dim_tuple(
            input_tensor, combine_fn, param_count, dim, other, keep_dims
        )

    # Handle single tensor inputs
    if dim is None:
        return _reduce_all_dims_single(
            input_tensor, combine_fn, param_count, cast("float", other), keep_dims
        )
    return _reduce_specific_dim_single(
        input_tensor, combine_fn, param_count, dim, cast("float", other), keep_dims
    )


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


@_decorators.ref(_reduce)
def _(
    combine_graph_id: int,
    input_tensor: torch.Tensor | tuple[torch.Tensor, ...],
    dim: int | None = None,
    keep_dims: bool = False,
    is_tuple_input: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    # For ref mode, we don't have access to the combine graph
    # This should be handled by the higher-level reduce ref implementation
    raise NotImplementedError("_reduce should not be called in ref mode")


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
        expr_from_string(f"reduce_result[{i}]", reduce_result=reduce_expr)
        for i in range(num_elements)
    ]
