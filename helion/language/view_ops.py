from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch

from .. import exc
from .._compiler.ast_extension import expr_from_string
from .._compiler.compile_environment import CompileEnvironment
from ..exc import NotInsideKernel
from . import _decorators

if TYPE_CHECKING:
    import ast

    from .._compiler.inductor_lowering import CodegenState

__all__ = ["join", "split", "subscript"]


def _split_dim(tensor: torch.Tensor, dim: int, op: str) -> tuple[int, int]:
    if type(dim) is not int or not -tensor.ndim <= dim < tensor.ndim:
        raise exc.UnsupportedSplitConfiguration(
            op=op, requirement="a constant dim within the input rank"
        )
    dim %= tensor.ndim
    size = tensor.shape[dim]
    if isinstance(size, torch.SymInt):
        env = CompileEnvironment.current()
        expr = env.specialize_expr(env.shape_env.simplify(size._sympy_()))
        block_id = env.resolve_block_id(size)
        if not isinstance(expr, sympy.Integer) and block_id is not None:
            block = env.block_sizes[block_id]
            if block.reduction:
                # A full-axis load has a block symbol even when its logical
                # extent is constant. Do not use the extent of a tiled axis.
                expr = env.specialize_expr(env.shape_env.simplify(block.numel))
        if not isinstance(expr, sympy.Integer):
            raise exc.UnsupportedSplitConfiguration(
                op=op, requirement="a compile-time constant split dimension size"
            )
        size = int(expr)
    return dim, size


def _unbind_two(
    tensor: torch.Tensor, dim: int, op: str
) -> tuple[torch.Tensor, torch.Tensor]:
    env = CompileEnvironment.current()
    if env.backend_name != "triton":
        raise exc.BackendUnsupported(env.backend_name, f"{op} device lowering")
    if dim != tensor.ndim - 1:
        order = [i for i in range(tensor.ndim) if i != dim] + [dim]
        tensor = tensor.permute(order)
    return split(tensor)


@_decorators.device_func_replacement(torch.unbind)
@_decorators.device_func_replacement(torch.Tensor.unbind)
def _torch_unbind(
    input: torch.Tensor,  # noqa: A002  Match PyTorch's input keyword.
    dim: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lower a size-two unbind to a permutation and hl.split."""
    dim, size = _split_dim(input, dim, "torch.unbind")
    if size != 2:
        raise exc.UnsupportedSplitConfiguration(
            op="torch.unbind", requirement="a split dimension of size 2"
        )
    shape = list(input.shape)
    shape[dim] = size
    return _unbind_two(input.reshape(shape), dim, "torch.unbind")


@_decorators.device_func_replacement(torch.chunk)
@_decorators.device_func_replacement(torch.Tensor.chunk)
def _torch_chunk(
    input: torch.Tensor,  # noqa: A002  Match PyTorch's input keyword.
    chunks: int,
    dim: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lower two equal, contiguous chunks through a size-two unbind."""
    if type(chunks) is not int or chunks != 2:
        raise exc.UnsupportedSplitConfiguration(
            op="torch.chunk", requirement="chunks=2"
        )
    dim, size = _split_dim(input, dim, "torch.chunk")
    if size == 0 or size % 2:
        raise exc.UnsupportedSplitConfiguration(
            op="torch.chunk", requirement="a positive even split dimension size"
        )
    if CompileEnvironment.current().backend.pad_factory_tensors_to_power_of_2 and (
        size & (size - 1)
    ):
        # Reshaping a padded axis into [2, size // 2] would split at the
        # padded midpoint instead of the logical midpoint.
        raise exc.UnsupportedSplitConfiguration(
            op="torch.chunk",
            requirement="a power-of-two split dimension size on this backend",
        )
    shape = list(input.shape)
    shape[dim : dim + 1] = [2, size // 2]
    return _unbind_two(input.reshape(shape), dim, "torch.chunk")


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
        - Supports None and : (slice(None)) indexing on every backend
        - Used for reshaping kernel tensors by adding dimensions
        - Prefer direct indexing syntax when possible: ``tensor[None, :]``
        - Pallas also supports one contiguous narrowing index when compiler
          planning can keep the source block resident in VMEM
    """
    raise NotInsideKernel


@_decorators.register_fake(subscript)
def _(tensor: torch.Tensor, index: list[object]) -> torch.Tensor:
    env = CompileEnvironment.current()
    output_size = env.backend.fake_subscript_shape(tensor, index)
    return env.new_index_result(tensor, output_size)


@_decorators.codegen(subscript, "common")
def _(state: CodegenState) -> ast.AST:
    output_keys = []
    # pyrefly: ignore [not-iterable]
    for val in state.proxy_arg(1):
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
    # pyrefly: ignore [bad-index]
    return tensor[indices]


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


@_decorators.ref(join)
def _(tensor0: torch.Tensor, tensor1: torch.Tensor) -> torch.Tensor:
    left, right = torch.broadcast_tensors(tensor0, tensor1)
    return torch.stack((left, right), dim=-1)
