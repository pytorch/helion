from __future__ import annotations

import ast
import operator
from typing import TYPE_CHECKING
from typing import Callable
from typing import Iterator
from typing import cast
from typing import overload

import torch
import torch._higher_order_ops as higher_order_ops
from torch.fx.experimental import proxy_tensor

from .. import exc
from . import _decorators

if TYPE_CHECKING:
    from .._compiler.helper_function import CombineFunction
    from .._compiler.inductor_lowering import CodegenState
    from .._compiler.type_propagation import Origin
    from .._compiler.type_propagation import TypeInfo


__all__ = ["associative_scan", "cumprod", "cumsum"]


# Helper functions for ref mode implementations
def _build_indices(
    shape: tuple[int, ...], dim: int, idx: int
) -> tuple[slice | int, ...]:
    """Build indexing tuple for accessing position idx along dimension dim."""
    indices: list[slice | int] = [slice(None)] * len(shape)
    indices[dim] = idx
    return tuple(indices)


def _iterate_scan_dimension(
    scan_size: int, reverse: bool
) -> Iterator[tuple[int, int, bool]]:
    """
    Generate iteration indices for scan operation.

    Yields:
        Tuple of (iteration_index, actual_index, is_first_element)
    """
    for i in range(scan_size):
        # Calculate current index based on scan direction
        idx = (scan_size - 1 - i) if reverse else i

        # Check if this is the first element in the scan
        is_first = (i == 0 and not reverse) or (i == scan_size - 1 and reverse)

        yield i, idx, is_first


def _get_prev_index(idx: int, reverse: bool) -> int:
    """Get the previous index in the scan sequence."""
    return (idx + 1) if reverse else (idx - 1)


def _scan_single_tensor(
    combine_fn: Callable, input_tensor: torch.Tensor, dim: int, reverse: bool
) -> torch.Tensor:
    """Helper function to perform scan on a single tensor."""
    result = torch.empty_like(input_tensor)
    scan_size = input_tensor.shape[dim]

    # Iterate through the dimension to scan
    for _i, idx, is_first in _iterate_scan_dimension(scan_size, reverse):
        # Build indexing tuple to access elements at position idx along dim
        indices = _build_indices(input_tensor.shape, dim, idx)

        if is_first:
            # First element: copy input directly
            result[indices] = input_tensor[indices]
        else:
            # Combine with previous accumulated value
            prev_idx = _get_prev_index(idx, reverse)
            prev_indices = _build_indices(input_tensor.shape, dim, prev_idx)

            # Apply the combine function
            result[indices] = combine_fn(result[prev_indices], input_tensor[indices])

    return result


def _scan_tuple_tensors(
    combine_fn: Callable, input_tuple: tuple[torch.Tensor, ...], dim: int, reverse: bool
) -> tuple[torch.Tensor, ...]:
    """Helper function to perform scan on a tuple of tensors."""
    tensors = list(input_tuple)
    scan_size = tensors[0].shape[dim]

    # Initialize result tensors
    results = [torch.empty_like(t) for t in tensors]

    # Iterate through the dimension to scan
    for _i, idx, is_first in _iterate_scan_dimension(scan_size, reverse):
        # Build indexing tuple
        indices = _build_indices(tensors[0].shape, dim, idx)

        if is_first:
            # First element: copy inputs directly
            for j, tensor in enumerate(tensors):
                results[j][indices] = tensor[indices]
        else:
            # Combine with previous accumulated values
            prev_idx = _get_prev_index(idx, reverse)
            prev_indices = _build_indices(tensors[0].shape, dim, prev_idx)

            # Gather values for combination
            current_vals = tuple(t[indices] for t in tensors)
            prev_vals = tuple(r[prev_indices] for r in results)

            # Apply combine function with unpacked arguments
            combined = combine_fn(*prev_vals, *current_vals)

            # Store results (handle both single and tuple returns)
            if isinstance(combined, tuple):
                for j, val in enumerate(combined):
                    results[j][indices] = val
            else:
                # Single result case
                results[0][indices] = combined

    return tuple(results)


@overload
@_decorators.device_func_replacement(higher_order_ops.associative_scan)
@_decorators.api(is_device_only=True)
def associative_scan(
    combine_fn: CombineFunction,
    input_tensor: torch.Tensor,
    dim: int,
    reverse: bool = False,
) -> torch.Tensor: ...


@overload
@_decorators.device_func_replacement(higher_order_ops.associative_scan)
@_decorators.api(is_device_only=True)
def associative_scan(
    combine_fn: CombineFunction,
    input_tensor: tuple[torch.Tensor, ...],
    dim: int,
    reverse: bool = False,
) -> tuple[torch.Tensor, ...]: ...


@_decorators.device_func_replacement(higher_order_ops.associative_scan)
@_decorators.api(is_device_only=True)
def associative_scan(
    combine_fn: CombineFunction,
    input_tensor: torch.Tensor | tuple[torch.Tensor, ...],
    dim: int,
    reverse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """
    Applies an associative scan operation along a specified dimension.

    Computes the prefix scan (cumulative operation) along a dimension using
    a custom combine function. Unlike :func:`~helion.language.reduce`, this
    preserves the input shape.

    Args:
        combine_fn: A binary function that combines two elements element-wise.
                   Must be associative for correct results.
                   Can be tensor->tensor or tuple->tuple function.
        input_tensor: Input tensor or tuple of tensors to scan
        dim: The dimension along which to scan
        reverse: If True, performs the scan in reverse order

    Returns:
        torch.Tensor or tuple[torch.Tensor, ...]: Tensor(s) with same shape as input
                                                  containing the scan result

    See Also:
        - :func:`~helion.language.reduce`: For dimension-reducing operations
        - :func:`~helion.language.cumsum`: For cumulative sum
        - :func:`~helion.language.cumprod`: For cumulative product

    Note:
        - combine_fn must be associative (not necessarily commutative)
        - Output has same shape as input (unlike reduce)
        - For standard scans, use :func:`~helion.language.cumsum` or :func:`~helion.language.cumprod` (faster)
        - Reverse scan applies the operation from right to left
    """
    raise exc.NotInsideKernel


@_decorators.register_fake(associative_scan)
def _(
    combine_fn: CombineFunction,
    input_tensor: torch.Tensor | tuple[torch.Tensor, ...],
    dim: int,
    reverse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Fake implementation that returns fake tensors with the same shape as input."""
    if isinstance(input_tensor, (tuple, list)):
        return tuple(torch.empty_like(t) for t in input_tensor)
    return torch.empty_like(input_tensor)


@_decorators.register_to_device_ir(associative_scan)
def _(
    tracer: proxy_tensor.PythonKeyTracer,
    combine_fn: CombineFunction,
    input_tensor: torch.Tensor | tuple[torch.Tensor, ...],
    dim: int,
    reverse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """
    Device IR implementation that handles tracing for associative_scan.  We map
    associative_scan to _associative_scan, with a pre-traced graph for the combine
    function.
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
            "associative_scan input must be a tuple of tensors"
        )
    else:
        assert isinstance(input_tensor, torch.Tensor), (
            "associative_scan input must be a tensor"
        )
    assert isinstance(dim, int), "associative_scan dim must be an integer"

    assert callable(combine_fn), "combine_fn must be callable"
    # Extract the function name before wrapping
    original_function_name = extract_helper_function_name(combine_fn)
    combine_fn = create_combine_function_wrapper(
        combine_fn, is_tuple_input=is_tuple_input, target_format="unpacked"
    )

    # Create fake inputs for the combine function
    fake_inputs = []
    for tensor in input_tensor if is_tuple_input else [input_tensor]:
        fake_inputs.extend(
            [
                torch.empty([1], dtype=tensor.dtype, device=tensor.device),
                torch.empty([1], dtype=tensor.dtype, device=tensor.device),
            ]
        )

    combine_graph = proxy_tensor.make_fx(
        combine_fn, decomposition_table=select_decomp_table()
    )(*fake_inputs).graph
    combine_graph_id = DeviceIR.current().add_graph(
        combine_graph,
        HelperFunctionGraphInfo,
        node_args=[],
        original_function_name=original_function_name,
    )

    # Create the associative_scan tracing operation
    scan_args = (combine_graph_id, input_tensor, dim, reverse, is_tuple_input)
    proxy_args, proxy_kwargs = args_to_proxies(tracer, scan_args)
    proxy_out = tracer.create_proxy(
        "call_function",
        _associative_scan,
        proxy_args,
        proxy_kwargs,
    )

    # The output has the same shape as the input
    if is_tuple_input:
        proxy_tensor.track_tensor_tree(
            input_tensor, proxy_out, constant=None, tracer=tracer
        )
        tuple_proxies = []
        assert isinstance(input_tensor, (tuple, list))
        for i, tensor in enumerate(input_tensor):
            element_proxy = tracer.create_proxy(
                "call_function",
                operator.getitem,
                (proxy_out, i),
                {},
            )
            proxy_tensor.track_tensor_tree(
                tensor, element_proxy, constant=None, tracer=tracer
            )
            tuple_proxies.append(tensor)
        return tuple(tuple_proxies)

    proxy_tensor.track_tensor_tree(
        input_tensor, proxy_out, constant=None, tracer=tracer
    )
    return input_tensor


@_decorators.type_propagation(associative_scan)
def _(
    combine_fn: TypeInfo,
    input_tensor: TypeInfo,
    dim: TypeInfo,
    reverse: TypeInfo | None = None,
    *,
    origin: Origin,
) -> TypeInfo:
    """Type propagation for associative_scan - output has same type as input."""
    from .._compiler.type_propagation import CallableType
    from .._compiler.type_propagation import SequenceType
    from .._compiler.type_propagation import TensorType

    # Validate that combine_fn is callable
    if not isinstance(combine_fn, CallableType):
        raise exc.TypeInferenceError(f"combine_fn must be callable, got {combine_fn}")

    # Validate that input_tensor is a tensor or tuple of tensors
    if isinstance(input_tensor, TensorType):
        # Single tensor case
        return input_tensor
    if isinstance(input_tensor, SequenceType):
        # Tuple of tensors case - validate all elements are tensors
        for elem_type in input_tensor.unpack():
            if not isinstance(elem_type, TensorType):
                raise exc.TypeInferenceError(
                    f"All elements in tuple must be tensors, got {elem_type}"
                )
        # Return the same tuple type
        return input_tensor
    raise exc.TypeInferenceError(
        f"input_tensor must be a tensor or tuple of tensors, got {input_tensor}"
    )


@_decorators.ref(associative_scan)
def _(
    combine_fn: Callable,
    input_tensor: torch.Tensor | tuple[torch.Tensor, ...],
    dim: int,
    reverse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    if isinstance(input_tensor, (tuple, list)):
        return _scan_tuple_tensors(combine_fn, tuple(input_tensor), dim, reverse)
    return _scan_single_tensor(combine_fn, input_tensor, dim, reverse)


@_decorators.device_func_replacement(torch.cumsum)
def cumsum(input_tensor: torch.Tensor, dim: int, reverse: bool = False) -> torch.Tensor:
    """
    Compute the cumulative sum along a specified dimension.

    Equivalent to ``hl.associative_scan(torch.add, input_tensor, dim, reverse)``.

    Args:
        input_tensor: Input tensor to compute cumulative sum
        dim: The dimension along which to compute cumulative sum
        reverse: If True, performs the cumsum in reverse order

    Returns:
        torch.Tensor: Tensor with same shape as input containing cumulative sum

    See Also:
        - :func:`~helion.language.associative_scan`: For custom scan operations
        - :func:`~helion.language.cumprod`: For cumulative product
        - :func:`~helion.language.reduce`: For dimension-reducing operations

    Note:
        - Output has same shape as input
        - Reverse=True computes cumsum from right to left
        - Equivalent to torch.cumsum
    """
    return associative_scan(torch.add, input_tensor, dim, reverse)


@_decorators.device_func_replacement(torch.cumprod)
def cumprod(
    input_tensor: torch.Tensor, dim: int, reverse: bool = False
) -> torch.Tensor:
    """
    Compute the cumulative product along a specified dimension.

    Equivalent to ``hl.associative_scan(torch.mul, input_tensor, dim, reverse)``.

    Args:
        input_tensor: Input tensor to compute cumulative product
        dim: The dimension along which to compute cumulative product
        reverse: If True, performs the cumprod in reverse order

    Returns:
        torch.Tensor: Tensor with same shape as input containing cumulative product

    See Also:
        - :func:`~helion.language.associative_scan`: For custom scan operations
        - :func:`~helion.language.cumsum`: For cumulative sum
        - :func:`~helion.language.reduce`: For dimension-reducing operations

    Note:
        - Output has same shape as input
        - Reverse=True computes cumprod from right to left
        - Equivalent to torch.cumprod
    """
    return associative_scan(torch.mul, input_tensor, dim, reverse)


@_decorators.api()
def _associative_scan(
    combine_graph_id: int,
    input_tensor: torch.Tensor | tuple[torch.Tensor, ...],
    dim: int,
    reverse: bool = False,
    is_tuple_input: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Device IR implementation of associative scan, not meant to be called directly."""
    raise AssertionError("this should never be called")


@_decorators.register_fake(_associative_scan)
def _(
    combine_graph_id: int,
    input_tensor: torch.Tensor | tuple[torch.Tensor, ...],
    dim: int,
    reverse: bool = False,
    is_tuple_input: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Fake implementation that returns a tensor/tuple with the same shape as input."""
    if is_tuple_input:
        assert isinstance(input_tensor, (tuple, list)), input_tensor
        return tuple(torch.empty_like(t) for t in input_tensor)
    assert isinstance(input_tensor, torch.Tensor), input_tensor
    return torch.empty_like(input_tensor)


@_decorators.codegen(_associative_scan)
def _(state: CodegenState) -> ast.AST | list[ast.AST]:
    """Generate code for associative scan with combine function."""

    combine_graph_id = state.proxy_arg(0)
    dim = state.proxy_arg(2)
    reverse = state.proxy_arg(3)
    is_tuple_input = state.proxy_arg(4)

    input_tensor = _get_input_tensor_ast(state, bool(is_tuple_input))
    helper_func_name = _register_helper_function(state, cast("int", combine_graph_id))
    scan_expr = _create_scan_expression(
        input_tensor, cast("int", dim), helper_func_name, bool(reverse)
    )

    if is_tuple_input:
        return _create_tuple_result_expressions(state, scan_expr)
    return scan_expr


@_decorators.ref(_associative_scan)
def _(
    combine_graph_id: int,
    input_tensor: torch.Tensor | tuple[torch.Tensor, ...],
    dim: int,
    reverse: bool = False,
    is_tuple_input: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    # For ref mode, we don't have access to the combine graph
    # This should be handled by the higher-level associative_scan ref implementation
    raise NotImplementedError("_associative_scan should not be called in ref mode")


def _get_input_tensor_ast(state: CodegenState, is_tuple_input: bool) -> ast.AST:
    """Get the input tensor AST, handling tuple inputs specially."""
    if not is_tuple_input:
        return state.ast_arg(1)

    raw_input = state.ast_args[1]
    if isinstance(raw_input, tuple):
        from .._compiler.ast_extension import create

        tuple_elts = [
            elt if isinstance(elt, ast.AST) else ast.Constant(value=elt)
            for elt in raw_input
        ]
        return create(ast.Tuple, elts=tuple_elts, ctx=ast.Load())
    return state.ast_arg(1)


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


def _create_scan_expression(
    input_tensor: ast.AST, dim: int, helper_func_name: str, reverse: bool
) -> ast.AST:
    """Create the tl.associative_scan expression."""
    from .._compiler.ast_extension import expr_from_string

    template = (
        f"tl.associative_scan(input_tensor, dim_value, {helper_func_name}, reverse=True)"
        if reverse
        else f"tl.associative_scan(input_tensor, dim_value, {helper_func_name})"
    )
    return expr_from_string(
        template,
        input_tensor=input_tensor,
        dim_value=ast.Constant(value=dim),
    )


def _create_tuple_result_expressions(
    state: CodegenState, scan_expr: ast.AST
) -> list[ast.AST]:
    """Create getitem expressions for tuple results."""
    from .._compiler.ast_extension import expr_from_string

    raw_input = state.ast_args[1]
    num_elements = len(raw_input) if isinstance(raw_input, tuple) else 2

    return [
        expr_from_string(f"scan_result[{i}]", scan_result=scan_expr)
        for i in range(num_elements)
    ]
