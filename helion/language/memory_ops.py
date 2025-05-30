from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch
from torch.fx import has_side_effect

from .. import exc
from .._compiler.indexing_strategy import SubscriptIndexing
from . import _decorators

if TYPE_CHECKING:
    from .._compiler.inductor_lowering import CodegenState

__all__ = ["atomic_add", "load", "store"]


@has_side_effect
@_decorators.api(tiles_as_sizes=True)
def store(
    tensor: torch.Tensor,
    index: list[object],
    value: torch.Tensor,
    extra_mask: torch.Tensor | None = None,
) -> None:
    """Store a value from to tensor using a list of indices.

    Args:
        tensor: The tensor to load from
        index: The indices to use to index into the tensor
        value: The value to store
        extra_mask: The extra mask (beyond automatic tile bounds masking) to apply to the tensor
    Returns:
        torch.Tensor: The loaded value
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(store)
def _(
    tensor: torch.Tensor,
    index: list[object],
    value: torch.Tensor,
    extra_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[object], torch.Tensor]:
    from helion._compiler.tile_index_proxy import TileIndexProxy

    if value.dtype != tensor.dtype:
        value = value.to(tensor.dtype)
    index = TileIndexProxy.tiles_to_sizes(index)
    return (tensor, index, value, extra_mask)


@_decorators.register_fake(store)
def _(
    tensor: torch.Tensor,
    index: list[object],
    value: torch.Tensor,
    extra_mask: torch.Tensor | None = None,
) -> None:
    return None


@_decorators.codegen(store)
def _(state: CodegenState) -> ast.AST:
    tensor = state.proxy_arg(0)
    assert isinstance(tensor, torch.Tensor)
    subscript = state.proxy_arg(1)
    assert isinstance(subscript, (list, tuple))
    value = state.ast_arg(2)
    extra_mask = state.ast_args[3]
    assert isinstance(extra_mask, (type(None), ast.AST))
    return state.device_function.indexing_strategy.codegen_store(
        state, tensor, [*subscript], value, extra_mask
    )


@_decorators.api(tiles_as_sizes=True)
def load(
    tensor: torch.Tensor, index: list[object], extra_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Load a value from a tensor using a list of indices.

    Args:
        tensor: The tensor to load from
        index: The indices to use to index into the tensor
        extra_mask: The extra mask (beyond automatic tile bounds masking) to apply to the tensor
    Returns:
        torch.Tensor: The loaded value
    """
    raise exc.NotInsideKernel


@_decorators.register_fake(load)
def _(
    tensor: torch.Tensor, index: list[object], extra_mask: torch.Tensor | None = None
) -> torch.Tensor:
    return tensor.new_empty(SubscriptIndexing.compute_shape(tensor, index))


@_decorators.codegen(load)
def _(state: CodegenState) -> ast.AST:
    tensor = state.proxy_arg(0)
    assert isinstance(tensor, torch.Tensor)
    subscript = state.proxy_arg(1)
    assert isinstance(subscript, (list, tuple))
    extra_mask = state.ast_args[2]
    assert isinstance(extra_mask, (type(None), ast.AST))
    return state.device_function.indexing_strategy.codegen_load(
        state, tensor, [*subscript], extra_mask
    )


@has_side_effect
@_decorators.api()
def atomic_add(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | float,
    sem: str = "relaxed",
) -> None:
    """
    Atomically add a value to a target tensor.

    Args:
        target: The tensor to add to
        index: Indices into target for way to accumulate values
        value: The value to add
        sem: The memory ordering semantics (default: 'relaxed')

    Returns:
        None
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(atomic_add)
def _(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | float,
    sem: str = "relaxed",
) -> tuple[torch.Tensor, object, torch.Tensor | float | int, str]:
    from helion._compiler.tile_index_proxy import TileIndexProxy

    valid_sems = {"relaxed", "acquire", "release", "acq_rel"}
    if sem not in valid_sems:
        raise ValueError(
            f"Invalid memory semantic '{sem}'. Must be one of {valid_sems}."
        )

    index = TileIndexProxy.prepare_index(index)
    index = TileIndexProxy.tiles_to_sizes(index)

    return (target, index, value, sem)


@_decorators.register_fake(atomic_add)
def _(
    target: torch.Tensor, index: list[object], value: torch.Tensor, sem: str = "relaxed"
) -> None:
    return None


@_decorators.codegen(atomic_add)
def _(state: CodegenState) -> ast.AST:
    import ast

    from .._compiler.ast_extension import expr_from_string

    target = state.proxy_arg(0)
    index = state.proxy_arg(1)
    value = state.proxy_arg(2)
    sem = expr_from_string(f"'{state.proxy_arg(3)}'")

    assert isinstance(target, torch.Tensor)
    assert isinstance(index, (list))

    indices = SubscriptIndexing.create(state, target, index)
    name = state.device_function.tensor_arg(target).name

    value_expr = (
        state.ast_args[2]
        if isinstance(value, torch.Tensor)
        else ast.Constant(value=value)
    )
    assert isinstance(value_expr, ast.AST)
    return expr_from_string(
        f"tl.atomic_add({name} + offset, value, mask=mask, sem=sem)",
        value=value_expr,
        offset=indices.index_expr,
        mask=indices.mask_expr,
        sem=sem,
    )
