from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch
from torch._inductor.codegen.simd import constant_repr
from torch.fx import has_side_effect

from .. import exc
from .._compiler.ast_extension import expr_from_string
from .._compiler.indexing_strategy import SubscriptIndexing
from . import _decorators

if TYPE_CHECKING:
    from .._compiler.inductor_lowering import CodegenState

__all__ = ["atomic_add", "load", "store"]


@has_side_effect
@_decorators.api(tiles_as_sizes=True, allow_host_tensor=True)
def store(
    tensor: torch.Tensor,
    index: list[object],
    value: torch.Tensor | torch.SymInt | float,
    extra_mask: torch.Tensor | None = None,
) -> None:
    """Store a value to a tensor using a list of indices.

    This function is equivalent to `tensor[index] = value` but allows
    setting `extra_mask=` to mask elements beyond the default masking
    based on the hl.tile range.

    Args:
        tensor: The tensor to store to
        index: The indices to use to index into the tensor
        value: The value to store
        extra_mask: The extra mask (beyond automatic tile bounds masking) to apply to the tensor
    Returns:
        None
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(store)
def _(
    tensor: torch.Tensor,
    index: list[object],
    value: torch.Tensor | torch.SymInt | float,
    extra_mask: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor, list[object], torch.Tensor | torch.SymInt | float, torch.Tensor | None
]:
    from .tile_proxy import Tile

    if isinstance(value, torch.Tensor) and value.dtype != tensor.dtype:
        value = value.to(tensor.dtype)
    index = Tile._tiles_to_sizes(index)
    return (tensor, index, value, extra_mask)


@_decorators.register_fake(store)
def _(
    tensor: torch.Tensor,
    index: list[object],
    value: torch.Tensor | torch.SymInt | float,
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


@_decorators.api(tiles_as_sizes=True, allow_host_tensor=True)
def load(
    tensor: torch.Tensor, index: list[object], extra_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Load a value from a tensor using a list of indices.

    This function is equivalent to `tensor[index]` but allows
    setting `extra_mask=` to mask elements beyond the default masking
    based on the hl.tile range.

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


@_decorators.get_masked_value(load)
def _(node: torch.fx.Node) -> int:
    return 0  # loads are always masked to 0


@has_side_effect
@_decorators.api(allow_host_tensor=True)
def atomic_add(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | float,
    sem: str = "relaxed",
) -> None:
    """
    Atomically add a value to a target tensor.

    Performs an atomic read-modify-write operation that adds value to target[index].
    This is safe for concurrent access from multiple threads/blocks.

    Args:
        target: The tensor to add to
        index: Indices into target for accumulating values
        value: The value to add (tensor or scalar)
        sem: Memory ordering semantics (default: 'relaxed')
            - 'relaxed': No ordering constraints
            - 'acquire': Acquire semantics
            - 'release': Release semantics
            - 'acq_rel': Acquire-release semantics

    Returns:
        None

    Examples:
        .. code-block:: python

            @helion.kernel
            def global_sum(x: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
                # Each tile computes local sum, then atomically adds to global
                for tile in hl.tile(x.size(0)):
                    local_data = x[tile]
                    local_sum = local_data.sum()
                    hl.atomic_add(result, [0], local_sum)

                return result

    See Also:
        - :func:`~helion.language.store`: For non-atomic stores
        - :func:`~helion.language.load`: For atomic loads

    Note:
        - Required for race-free accumulation across parallel execution
        - Performance depends on memory access patterns and contention
        - Consider using regular operations when atomicity isn't needed
        - Higher memory semantics (acquire/release) have performance overhead
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(atomic_add)
def _(
    target: torch.Tensor,
    index: list[object],
    value: torch.Tensor | float,
    sem: str = "relaxed",
) -> tuple[torch.Tensor, object, torch.Tensor | float | int, str]:
    from .tile_proxy import Tile

    valid_sems = {"relaxed", "acquire", "release", "acq_rel"}
    if sem not in valid_sems:
        raise ValueError(
            f"Invalid memory semantic '{sem}'. Must be one of {valid_sems}."
        )

    index = Tile._prepare_index(index)
    index = Tile._tiles_to_sizes(index)

    return (target, index, value, sem)


@_decorators.register_fake(atomic_add)
def _(
    target: torch.Tensor, index: list[object], value: torch.Tensor, sem: str = "relaxed"
) -> None:
    return None


@_decorators.codegen(atomic_add)
def _(state: CodegenState) -> ast.AST:
    target = state.proxy_arg(0)
    index = state.proxy_arg(1)
    sem = expr_from_string(repr(state.proxy_arg(3)))

    assert isinstance(target, torch.Tensor)
    assert isinstance(index, list)

    indices = SubscriptIndexing.create(state, target, index)
    name = state.device_function.tensor_arg(target).name

    value_expr = state.ast_args[2]
    if isinstance(value_expr, (int, float, bool)):
        value_expr = expr_from_string(constant_repr(value_expr))
    assert isinstance(value_expr, ast.AST)
    return expr_from_string(
        f"tl.atomic_add({name} + offset, value, mask=mask, sem=sem)",
        value=value_expr,
        offset=indices.index_expr,
        mask=indices.mask_expr,
        sem=sem,
    )
