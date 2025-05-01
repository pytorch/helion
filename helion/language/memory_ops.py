from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.fx import has_side_effect

from .. import exc
from .._compiler.indexing_strategy import SubscriptIndexing
from . import _decorators

if TYPE_CHECKING:
    import ast

    from .._compiler.inductor_lowering import CodegenState

__all__ = ["load", "store"]


@has_side_effect
@_decorators.api(tiles_as_sizes=True)
def store(tensor: torch.Tensor, index: list[object], value: torch.Tensor) -> None:
    raise exc.NotInsideKernel


@_decorators.prepare_args(store)
def _(
    tensor: torch.Tensor, index: list[object], value: torch.Tensor
) -> tuple[torch.Tensor, list[object], torch.Tensor]:
    from helion._compiler.tile_index_proxy import TileIndexProxy

    if value.dtype != tensor.dtype:
        value = value.to(tensor.dtype)
    index = TileIndexProxy.tiles_to_sizes(index)
    return (tensor, index, value)


@_decorators.register_fake(store)
def _(tensor: torch.Tensor, index: list[object], value: torch.Tensor) -> None:
    return None


@_decorators.codegen(store)
def _(state: CodegenState) -> ast.AST:
    tensor = state.proxy_arg(0)
    assert isinstance(tensor, torch.Tensor)
    subscript = state.proxy_arg(1)
    assert isinstance(subscript, (list, tuple))
    return state.device_function.indexing_strategy.codegen_store(
        state, tensor, [*subscript]
    )


@_decorators.api(tiles_as_sizes=True)
def load(tensor: torch.Tensor, index: list[object]) -> torch.Tensor:
    raise exc.NotInsideKernel


@_decorators.register_fake(load)
def _(tensor: torch.Tensor, index: list[object]) -> torch.Tensor:
    return tensor.new_empty(SubscriptIndexing.compute_shape(tensor, index))


@_decorators.codegen(load)
def _(state: CodegenState) -> ast.AST:
    tensor = state.proxy_arg(0)
    assert isinstance(tensor, torch.Tensor)
    subscript = state.proxy_arg(1)
    assert isinstance(subscript, (list, tuple))
    return state.device_function.indexing_strategy.codegen_load(
        state, tensor, [*subscript]
    )
