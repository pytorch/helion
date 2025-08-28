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
        else:
            raise exc.InvalidIndexingType(repr(val))
    assert len(input_size) == 0
    return tensor.new_empty(output_size)


@_decorators.codegen(subscript)
def _(state: CodegenState) -> ast.AST:
    output_keys = []
    for val in state.proxy_arg(1):  # pyright: ignore[reportGeneralTypeIssues]
        if val is None:
            output_keys.append("None")
        elif isinstance(val, slice) and repr(val) == "slice(None, None, None)":
            output_keys.append(":")
        else:
            raise exc.InvalidIndexingType(repr(val))
    return expr_from_string(
        f"{{base}}[{', '.join(output_keys)}]",
        base=state.ast_arg(0),
    )


@_decorators.ref(subscript)
def _(tensor: torch.Tensor, indices: list[object]) -> torch.Tensor:
    return tensor[indices]  # pyright: ignore[reportArgumentType]


@_decorators.get_masked_value(subscript)
def _(node: torch.fx.Node) -> float | bool | None:
    from .._compiler.node_masking import cached_masked_value

    other = node.args[0]
    assert isinstance(other, torch.fx.Node)
    return cached_masked_value(other)
