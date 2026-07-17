from __future__ import annotations

import operator
from typing import TYPE_CHECKING
from typing import cast
from typing import overload

import torch
import torch._higher_order_ops as higher_order_ops
from torch.fx.experimental import proxy_tensor

from .. import exc
from . import _decorators

if TYPE_CHECKING:
    from .._compiler.helper_function import CombineFunction
    from .._compiler.helper_function import CombineFunctionBasic
    from .._compiler.helper_function import CombineFunctionTuple
    from .._compiler.type_info import Origin
    from .._compiler.type_info import TypeInfo


__all__ = ["associative_scan", "cumprod", "cumsum"]


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


@_decorators.ref(associative_scan)
def _(
    combine_fn: CombineFunction,
    input_tensor: torch.Tensor | tuple[torch.Tensor, ...],
    dim: int,
    reverse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    # Eager inclusive scan. Ref mode must not call torch's associative_scan HOP: it
    # runs Dynamo, which breaks under Helion's active RefMode TorchFunctionMode.
    from .._compiler.helper_function import create_combine_function_wrapper

    is_tuple = isinstance(input_tensor, (tuple, list))
    leaves = tuple(input_tensor) if is_tuple else (input_tensor,)
    combine = create_combine_function_wrapper(
        combine_fn, is_tuple_input=is_tuple, target_format="tuple"
    )

    scan_dim = dim % leaves[0].ndim
    length = leaves[0].size(scan_dim)
    order = reversed(range(length)) if reverse else range(length)

    acc: tuple[torch.Tensor, ...] | None = None
    out: list[tuple[torch.Tensor, ...]] = [()] * length
    for k in order:
        cur = tuple(leaf.select(scan_dim, k) for leaf in leaves)
        if acc is None:
            acc = cur
        elif is_tuple:
            acc = tuple(cast("CombineFunctionTuple", combine)(acc, cur))
        else:
            acc = (cast("CombineFunctionBasic", combine)(acc[0], cur[0]),)
        out[k] = acc

    result = tuple(torch.stack(col, dim=scan_dim) for col in zip(*out, strict=True))
    return result if is_tuple else result[0]


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

    # Create new output tensors to avoid aliasing input with output.
    # tl.associative_scan modifies its input in-place, so we must track
    # distinct output tensors to ensure the input remains usable after the scan.
    if is_tuple_input:
        output_tensors = []
        assert isinstance(input_tensor, (tuple, list))
        for i, tensor in enumerate(input_tensor):
            output_tensor = torch.empty_like(tensor)
            element_proxy = tracer.create_proxy(
                "call_function",
                operator.getitem,
                (proxy_out, i),
                {},
            )
            proxy_tensor.track_tensor_tree(
                output_tensor, element_proxy, constant=None, tracer=tracer
            )
            output_tensors.append(output_tensor)
        return tuple(output_tensors)

    output_tensor = torch.empty_like(input_tensor)
    proxy_tensor.track_tensor_tree(
        output_tensor, proxy_out, constant=None, tracer=tracer
    )
    return output_tensor


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
    from .._compiler.type_info import CallableType
    from .._compiler.type_info import SequenceType
    from .._compiler.type_info import TensorType

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


# ---------------------------------------------------------------------------
# Backend-specific codegens for these ops live in per-backend modules under
# helion/_compiler/<backend>/.  Import them here (at module import time) so the
# @_decorators.codegen(op, "<backend>") registrations run with the same eager
# timing as when the bodies lived in this file -- no behavior change.
import helion._compiler.cute.scan_ops  # noqa: E402, F401
import helion._compiler.triton.scan_ops  # noqa: E402, F401
