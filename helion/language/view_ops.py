from __future__ import annotations

import collections
from typing import TYPE_CHECKING
from typing import cast

import torch

from .. import exc
from .._compiler.ast_extension import expr_from_string
from ..exc import NotInsideKernel
from . import _decorators

if TYPE_CHECKING:
    import ast

    from .._compiler.inductor_lowering import CodegenState

__all__ = ["join", "split", "subscript"]


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


@_decorators.api(is_device_only=True)
def split(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Split the last dimension of a tensor with size two into two separate tensors.

    Args:
        tensor: The input tensor whose last dimension has length two.

    Returns:
        A tuple ``(lo, hi)`` where each tensor has the same shape as ``tensor``
        without its last dimension.

    See Also:
        - :func:`~helion.language.join`
    """
    raise NotInsideKernel


@_decorators.register_fake(split)
def _(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out_shape = tensor.shape[:-1]
    return (
        tensor.new_empty(out_shape),
        tensor.new_empty(out_shape),
    )


@_decorators.codegen(split)
def _(state: CodegenState) -> list[ast.AST]:
    split_call = expr_from_string("tl.split({tensor})", tensor=state.ast_arg(0))
    return [
        expr_from_string("{value}[0]", value=split_call),
        expr_from_string("{value}[1]", value=split_call),
    ]


@_decorators.ref(split)
def _(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return cast("tuple[torch.Tensor, torch.Tensor]", torch.unbind(tensor, dim=-1))


@_decorators.api(is_device_only=True)
def join(
    tensor0: torch.Tensor,
    tensor1: torch.Tensor,
) -> torch.Tensor:
    """
    Join two tensors along a new minor dimension.

    Args:
        tensor0: First tensor to join.
        tensor1: Second tensor to join. Must be broadcast-compatible with
            ``tensor0``.

    Returns:
        torch.Tensor: A tensor with shape ``broadcast_shape + (2,)`` where
        ``broadcast_shape`` is the broadcast of the input shapes.

    See Also:
        - :func:`~helion.language.split`
    """
    raise NotInsideKernel


@_decorators.register_fake(join)
def _(tensor0: torch.Tensor, tensor1: torch.Tensor) -> torch.Tensor:
    if tensor0.dtype != tensor1.dtype:
        raise TypeError("join() requires both tensors to have the same dtype")
    if tensor0.device != tensor1.device:
        raise ValueError("join() requires both tensors to be on the same device")

    broadcast_shape = torch.broadcast_shapes(tensor0.shape, tensor1.shape)
    return tensor0.new_empty([*broadcast_shape, 2])


@_decorators.codegen(join)
def _(state: CodegenState) -> ast.AST:
    return expr_from_string(
        "tl.join({tensor0}, {tensor1})",
        tensor0=state.ast_arg(0),
        tensor1=state.ast_arg(1),
    )


@_decorators.ref(join)
def _(tensor0: torch.Tensor, tensor1: torch.Tensor) -> torch.Tensor:
    left, right = torch.broadcast_tensors(tensor0, tensor1)
    return torch.stack((left, right), dim=-1)
