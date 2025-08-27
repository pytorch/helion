from __future__ import annotations

import collections
from typing import TYPE_CHECKING

import torch

from .. import exc
from .._compiler.ast_extension import expr_from_string
from ..exc import NotInsideKernel
from . import _decorators

if TYPE_CHECKING:
    import ast

    from .._compiler.inductor_lowering import CodegenState

__all__ = ["subscript"]


@_decorators.api(tiles_as_sizes=True)
def subscript(tensor: torch.Tensor, index: list[object]) -> torch.Tensor:
    """
    Equivalent to tensor[index] where tensor is a kernel-tensor (not a host-tensor).

    Can be used to add dimensions to the tensor, e.g. tensor[None, :] or tensor[:, None].

    Args:
        tensor: The kernel tensor to index
        index: List of indices, including None for new dimensions and : for existing dimensions

    Returns:
        torch.Tensor: The indexed tensor with potentially modified dimensions

    Examples:
        .. code-block:: python

            @helion.kernel
            def broadcast_multiply(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                # x has shape (N,), y has shape (M,)
                result = torch.empty(
                    [x.size(0), y.size(0)], dtype=x.dtype, device=x.device
                )

                for tile_i, tile_j in hl.tile([x.size(0), y.size(0)]):
                    # Get tile data
                    x_tile = x[tile_i]
                    y_tile = y[tile_j]

                    # Make x broadcastable: (tile_size, 1)
                    # same as hl.subscript(x_tile, [slice(None), None])
                    x_expanded = x_tile[:, None]
                    # Make y broadcastable: (1, tile_size)
                    # same as hl.subscript(y_tile, [None, slice(None)])
                    y_expanded = y_tile[None, :]

                    result[tile_i, tile_j] = x_expanded * y_expanded

                return result

    See Also:
        - :func:`~helion.language.load`: For loading tensor values
        - :func:`~helion.language.store`: For storing tensor values

    Note:
        - Only supports None and : (slice(None)) indexing
        - Used for reshaping kernel tensors by adding dimensions
        - Prefer direct indexing syntax when possible: ``tensor[None, :]``
        - Does not support integer indexing or slicing with start/stop
    """
    raise NotInsideKernel


@_decorators.register_fake(subscript)
def _(tensor: torch.Tensor, index: list[object]) -> torch.Tensor:
    input_size = collections.deque(tensor.size())
    output_size = []
    for val in index:
        if val is None:
            output_size.append(1)
        elif isinstance(val, slice) and repr(val) == "slice(None, None, None)":
            output_size.append(input_size.popleft())
        elif isinstance(val, torch.SymInt):
            # Handle tile indices (tiles are converted to SymInts)
            # When a tile is used as an index, it means "select all elements" like ':'
            output_size.append(input_size.popleft())
        elif isinstance(val, int):
            # Handle integer indices - they select a single element, removing that dimension
            input_size.popleft()  # Consume the dimension but don't add to output
        else:
            raise exc.InvalidIndexingType(repr(val))
    assert len(input_size) == 0
    
    # Create a view that shares the same underlying data as the input tensor
    # This is similar to how reshape/view operations work in PyTorch
    result = tensor.new_empty(output_size)
    
    # CRITICAL: Mark this tensor as a view of the original
    # This ensures it's not treated as a new tensor needing separate loading
    # We do this by sharing the exact same data_ptr in the fake tensor system
    if hasattr(tensor, '_fake_tensor_memo'):
        result._fake_tensor_memo = tensor._fake_tensor_memo
    
    # Also register with the same origin if the base tensor has one
    from .._compiler.host_function import HostFunction
    host_function = HostFunction.current()
    if tensor in host_function.tensor_to_origin:
        host_function.tensor_to_origin[result] = host_function.tensor_to_origin[tensor]
    
    return result


@_decorators.codegen(subscript)
def _(state: CodegenState) -> ast.AST:
    output_keys = []
    for val in state.proxy_arg(1):  # pyright: ignore[reportGeneralTypeIssues]
        if val is None:
            output_keys.append("None")
        elif isinstance(val, slice) and repr(val) == "slice(None, None, None)":
            output_keys.append(":")
        elif isinstance(val, torch.SymInt):
            # Handle tile indices (tiles are converted to SymInts)
            # When a tile is used as an index in subscript, it means "select all elements" like ':'
            output_keys.append(":")
        elif isinstance(val, int):
            # Handle integer indices
            output_keys.append(str(val))
        else:
            raise exc.InvalidIndexingType(repr(val))
    return expr_from_string(
        f"base[{', '.join(output_keys)}]",
        base=state.ast_arg(0),
    )


@_decorators.ref(subscript)
def _(tensor: torch.Tensor, indices: list[object]) -> torch.Tensor:
    # Convert any torch.SymInt back to regular indices for ref implementation
    processed_indices = []
    for idx in indices:
        if isinstance(idx, torch.SymInt):
            # For ref implementation, tiles act like full slices ':'
            processed_indices.append(slice(None))
        elif isinstance(idx, int):
            # Integer indices are passed through
            processed_indices.append(idx)
        else:
            processed_indices.append(idx)
    return tensor[tuple(processed_indices)]  # pyright: ignore[reportArgumentType]


@_decorators.get_masked_value(subscript)
def _(node: torch.fx.Node) -> float | bool | None:
    from .._compiler.node_masking import cached_masked_value

    other = node.args[0]
    assert isinstance(other, torch.fx.Node)
    return cached_masked_value(other)
