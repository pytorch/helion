from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch._inductor.utils import triton_type

from .._compiler.ast_extension import expr_from_string
from .._compiler.compile_environment import CompileEnvironment
from ..exc import NotInsideKernel
from . import _decorators
from .ref_tile import RefTile

if TYPE_CHECKING:
    import ast

    from .._compiler.inductor_lowering import CodegenState

__all__ = ["arange", "full", "zeros"]


def zeros(
    shape: list[object],
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Return a device-tensor filled with zeros.

    Equivalent to ``hl.full(shape, 0.0 if dtype.is_floating_point else 0, dtype=dtype)``.

    Note:
        Only use within ``hl.tile()`` loops for creating local tensors.
        For output tensor creation, use ``torch.zeros()`` with proper device placement.

    Args:
        shape: A list of sizes (or tile indices which are implicitly converted to sizes)
        dtype: Data type of the tensor (default: torch.float32)
        device: Device must match the current compile environment device

    Returns:
        torch.Tensor: A device tensor of the given shape and dtype filled with zeros

    Examples:

        .. code-block:: python

            @helion.kernel
            def process_kernel(input: torch.Tensor) -> torch.Tensor:
                result = torch.empty_like(input)

                for tile in hl.tile(input.size(0)):
                    buffer = hl.zeros([tile], dtype=input.dtype)  # Local buffer
                    buffer += input[tile]  # Add input values to buffer
                    result[tile] = buffer

                return result

    See Also:
        - :func:`~helion.language.full`: For filling with arbitrary values
        - :func:`~helion.language.arange`: For creating sequences
    """
    return full(
        shape, 0.0 if dtype.is_floating_point else 0, dtype=dtype, device=device
    )


@_decorators.api(tiles_as_sizes=True)
def full(
    shape: list[object],
    value: float,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Create a device-tensor filled with a specified value.

    Note:
        Only use within ``hl.tile()`` loops for creating local tensors.
        For output tensor creation, use ``torch.full()`` with proper device placement.

    Args:
        shape: A list of sizes (or tile indices which are implicitly converted to sizes)
        value: The value to fill the tensor with
        dtype: The data type of the tensor (default: torch.float32)
        device: Device must match the current compile environment device

    Returns:
        torch.Tensor: A device tensor of the given shape and dtype filled with value

    Examples:
        .. code-block:: python

            @helion.kernel
            def process_kernel(input: torch.Tensor) -> torch.Tensor:
                result = torch.empty_like(input)

                for tile in hl.tile(input.size(0)):
                    # Create local buffer filled with initial value
                    buffer = hl.full([tile], 0.0, dtype=input.dtype)
                    buffer += input[tile]  # Add input values to buffer
                    result[tile] = buffer

                return result

    See Also:
        - :func:`~helion.language.zeros`: For filling with zeros
        - :func:`~helion.language.arange`: For creating sequences
    """
    raise NotInsideKernel


@_decorators.register_fake(full)
def _full_fake(
    shape: list[int | torch.SymInt],
    value: float,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    if not isinstance(shape, (list, tuple)):
        raise TypeError(f"Expected list[SymInt], got {type(shape).__name__}")
    env = CompileEnvironment.current()
    env.add_kernel_tensor_size(shape)
    return torch.empty(
        [*shape],
        dtype=dtype,
        device=env.device if device is None else device,
    )


@_decorators.codegen(full)
def _full_codegen(state: CodegenState) -> ast.AST:
    fake_value = state.fake_value
    assert isinstance(fake_value, torch.Tensor)
    shape_str = state.device_function.tile_strategy.shape_str(fake_value.size())
    type_str = triton_type(fake_value.dtype)

    # Check if the value is static (literal) or dynamic (node)
    proxy_value = state.proxy_arg(1)
    if isinstance(proxy_value, (int, float, bool)):
        # For static values, use literal_expr to preserve special representations like float('-inf')
        value_str = state.device_function.literal_expr(proxy_value)
        return expr_from_string(f"tl.full({shape_str}, {value_str}, {type_str})")
    # For dynamic values, use ast_arg to get the proper AST representation
    value_ast = state.ast_arg(1)
    return expr_from_string(
        f"tl.full({shape_str}, {{value}}, {type_str})", value=value_ast
    )


@_decorators.get_masked_value(full)
def _(
    node: torch.fx.Node,
) -> float | bool | None:
    value = node.args[1]
    if isinstance(value, (int, float, bool)):
        return value
    # Return None for dynamic values (like tensor elements)
    return None


@_decorators.ref(full)
def _(
    shape: list[int | RefTile],
    value: float,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    processed_shape = []
    for s in shape:
        if isinstance(s, RefTile):
            processed_shape.append(s.end - s.begin)
        else:
            processed_shape.append(s)
    env = CompileEnvironment.current()
    return torch.full(
        processed_shape,
        value,
        dtype=dtype,
        device=env.device if device is None else device,
    )


def arange(
    *args: int,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    **kwargs: object,
) -> torch.Tensor:
    """
    Same as `torch.arange()`, but defaults to same device as the current kernel.

    Creates a 1D tensor containing a sequence of integers in the specified range,
    automatically using the current kernel's device and index dtype.

    Args:
        args: Positional arguments passed to torch.arange(start, end, step).
        dtype: Data type of the result tensor (defaults to kernel's index dtype)
        device: Device must match the current compile environment device
        kwargs: Additional keyword arguments passed to torch.arange

    Returns:
        torch.Tensor: 1D tensor containing the sequence

    See Also:
        - :func:`~helion.language.tile_index`: For getting tile indices
        - :func:`~helion.language.zeros`: For creating zero-filled tensors
        - :func:`~helion.language.full`: For creating constant-filled tensors
    """
    env = CompileEnvironment.current()
    if dtype is None:
        dtype = env.settings.index_dtype
    return torch.arange(
        *args,
        **kwargs,
        dtype=dtype,
        device=env.device if device is None else device,
    )
