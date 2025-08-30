from __future__ import annotations

import ast
import collections
from typing import TYPE_CHECKING

import torch

from .. import exc
from .._compiler.ast_extension import expr_from_string
from ..exc import NotInsideKernel
from . import _decorators

if TYPE_CHECKING:
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
    tensors = [v for v in index if isinstance(v, torch.Tensor)]
    if not tensors:
        sizes, output = collections.deque(tensor.size()), []
        for v in index:
            if v is None:
                output.append(1)
            elif isinstance(v, slice) and repr(v) == "slice(None, None, None)":
                output.append(sizes.popleft())
            else:
                raise exc.InvalidIndexingType(repr(v))
        return tensor.new_empty(output)
    
    if len(tensors) == tensor.ndim:
        return tensor.new_empty([t.shape[0] for t in tensors])
    elif len(tensors) == 1:
        return tensor.new_empty(list(tensors[0].shape) + list(tensor.shape[len(index):]))
    else:
        return tensor.new_empty(tensors[0].shape)


def _build_load_expr(state, base_ast, indices):
    if not (isinstance(base_ast, ast.Name) and base_ast.id.endswith('_tile')):
        return None
    
    ptr = base_ast.id[:-5]
    ndim = len(indices)
    strides = [f"{ptr}_stride_{i}" for i in range(ndim)]
    
    if hasattr(state, 'codegen') and (fn := getattr(state.codegen, 'device_function', None)):
        found_strides = [fn.tensor_stride(t, i).name 
                        for t, a in fn._tensor_args.items() 
                        if a.name == ptr 
                        for i in range(ndim)]
        if found_strides:
            strides = found_strides
    
    is_broadcast = (len(getattr(state, 'proxy_args', [])) > 1 and 
                   isinstance(state.proxy_args[1], list) and
                   all(isinstance(x, torch.Tensor) and x.ndim == 1 for x in state.proxy_args[1]))
    
    kwargs = {f"idx{i}": indices[i] for i in range(ndim)}
    
    if is_broadcast:
        slice_strs = [', '.join(':' if j == i else 'None' for j in range(ndim)) for i in range(ndim)]
        parts = [f"{{idx{i}}}[{slice_str}] * {strides[i]}" for i, slice_str in enumerate(slice_strs)]
    else:
        parts = [f"{{idx{i}}} * {strides[i]}" for i in range(ndim)]
    
    return expr_from_string(f"tl.load({ptr} + ({' + '.join(parts)}), None)", **kwargs)


def _is_full_slice(v):
    return isinstance(v, slice) and repr(v) == "slice(None, None, None)"


@_decorators.codegen(subscript)
def _(state: CodegenState) -> ast.AST:
    indices = state.proxy_arg(1)  # pyright: ignore[reportGeneralTypeIssues]
    
    # Check if any indices are tensors
    has_tensors = any(isinstance(v, torch.Tensor) for v in indices)
    
    if not has_tensors:
        # Basic indexing - no tensors
        keys = []
        for v in indices:
            if v is None:
                keys.append("None")
            elif _is_full_slice(v):
                keys.append(":")
            else:
                raise exc.InvalidIndexingType(repr(v))
        return expr_from_string(f"{{base}}[{', '.join(keys)}]", base=state.ast_arg(0))
    
    # Advanced indexing with tensors
    if not all(isinstance(idx, torch.Tensor) for idx in indices):
        raise exc.InvalidIndexingType("Mixed tensor/non-tensor indexing not supported")
    
    ast_indices = state.ast_arg(1) if isinstance(state.ast_arg(1), list) else [state.ast_arg(1)]  # pyright: ignore[reportGeneralTypeIssues]
    result = _build_load_expr(state, state.ast_arg(0), ast_indices)
    if not result:
        raise exc.InvalidIndexingType("Expected tensor variable ending with '_tile'")
    return result


@_decorators.ref(subscript)
def _(tensor: torch.Tensor, indices: list[object]) -> torch.Tensor:
    return tensor[indices]  # pyright: ignore[reportArgumentType]


@_decorators.get_masked_value(subscript)
def _(node: torch.fx.Node) -> float | bool | None:
    from .._compiler.node_masking import cached_masked_value

    other = node.args[0]
    assert isinstance(other, torch.fx.Node)
    return cached_masked_value(other)