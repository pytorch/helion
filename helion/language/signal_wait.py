from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch._inductor.utils import triton_type
from torch.fx import has_side_effect

from .. import exc
from .._compiler.indexing_strategy import SubscriptIndexing
from . import _decorators
from .stack_tensor import StackTensor

if TYPE_CHECKING:
    import ast

    from .._compiler.inductor_lowering import CodegenState

__all__ = ["signal", "wait"]


@has_side_effect
@_decorators.api(tiles_as_sizes=True, allow_host_tensor=True)
def wait(
    signal_pad: torch.Tensor | StackTensor,
    index: list[object] | None = None,
    signal: int = 1,
    update: int | None = None,
    scope: str = "gpu",
    hasSubsequentMemAccess: bool = True,
) -> None:
    """
    Wait for global memory barriers.

    Spins on global memory barriers until the signal values is observed on all barriers.

    Args:
        signal_pad: Tensor of global memory barriers to wait on
        index: Indices to index into the signal_pad tensor
        signal: the value to wait for
        update: Atomically update the signal_pad tensor with this value once the signal is observed. (default: None)
        scope: The scope of the lock (default: 'gpu')
        hasSubsequentMemAccess: Whether the wait is followed by a subsequence memory access (default: True)

    Returns:
        None
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(wait)
def _(
    signal_pad: torch.Tensor | StackTensor,
    index: list[object],
    signal: int = 1,
    update: int | None = None,
    scope: str = "gpu",
    hasSubsequentMemAccess: bool = True,
) -> tuple[torch.Tensor | tuple[object, ...], object, int, int | None, str, bool]:
    from .tile_proxy import Tile

    assert isinstance(signal_pad, (torch.Tensor, StackTensor))

    if signal_pad.dtype not in (torch.int32, torch.uint32):
        raise NotImplementedError(
            f"Unsupported signal pad dtype: {signal_pad.dtype}. Must be of torch.int32 or torch.uint32."
        )

    valid_scopes = {"sys", "gpu"}

    if scope not in valid_scopes:
        raise ValueError(f"Invalid scope '{scope}'. Must be one of {valid_scopes}.")

    index = Tile._prepare_index(index)
    index = Tile._tiles_to_sizes(index)

    if isinstance(signal_pad, StackTensor):
        return (tuple(signal_pad), index, signal, update, scope, hasSubsequentMemAccess)
    return (signal_pad, index, signal, update, scope, hasSubsequentMemAccess)


@_decorators.register_fake(wait)
def _(
    signal_pad: torch.Tensor | tuple[object, ...],
    index: list[object],
    signal: int = 1,
    update: int | None = None,
    scope: str = "gpu",
    hasSubsequentMemAccess: bool = True,
    as_ptrs: bool = False,
) -> None:
    return None


@_decorators.codegen(wait)
def _(state: CodegenState) -> ast.AST:
    import ast

    from .._compiler.ast_extension import expr_from_string
    from .._compiler.indexing_strategy import SubscriptIndexing

    signal_pad = state.proxy_arg(0)
    index = state.proxy_arg(1)
    signal = state.proxy_arg(2)
    update = state.proxy_arg(3)
    scope = state.proxy_arg(4)
    has_subsequent_load = state.proxy_arg(5)

    if isinstance(signal_pad, tuple):
        signal_pad = StackTensor(*signal_pad)

    assert isinstance(signal_pad, (torch.Tensor, StackTensor))
    assert isinstance(index, (list))

    assert type(scope) is str

    assert type(has_subsequent_load) is bool

    sem = "acquire" if has_subsequent_load else "relaxed"
    op = "atomic_cas" if update is not None else "ld"
    update = 0 if update is None else update
    skip_sync = not has_subsequent_load

    if isinstance(signal_pad, torch.Tensor):
        indices = SubscriptIndexing.create(state, signal_pad, index)
        shape = SubscriptIndexing.compute_shape(signal_pad, index)
        signal_pad_name = state.device_function.tensor_arg(signal_pad).name

        bar_addrs_expr = expr_from_string(
            f"{signal_pad_name} + {{offset}}", offset=indices.index_expr
        )
    elif isinstance(signal_pad, StackTensor):
        from .._compiler.indexing_strategy import StackIndexingStrategy

        subscript_shape = SubscriptIndexing.compute_shape(signal_pad.tensor_like, index)
        stack_shape = signal_pad.dev_ptrs.shape
        shape = subscript_shape + list(stack_shape)

        stack_broadcast, tensor_broadcast = StackIndexingStrategy.get_broadcast_str(
            stack_shape, subscript_shape
        )

        tensor_like_indices = SubscriptIndexing.create(
            state, signal_pad.tensor_like, index
        )

        dtype = triton_type(signal_pad.dtype)

        ast_tensors = state.ast_args[0]
        assert isinstance(ast_tensors, tuple)
        assert len(ast_tensors) == 2
        tensor_like_ast, dev_ptrs_ast = ast_tensors
        bar_addrs_expr = expr_from_string(
            f"{{base}}.to(tl.pointer_type({dtype})){stack_broadcast} + {{offset}}{tensor_broadcast}",
            base=dev_ptrs_ast,
            offset=tensor_like_indices.index_expr,
        )
    else:
        raise NotImplementedError(f"Unsupported signal pad type: {type(signal_pad)}")

    signal_expr = ast.Constant(value=signal)  # pyright: ignore[reportArgumentType]
    update_expr = ast.Constant(value=update)  # pyright: ignore[reportArgumentType]

    is_scalar = len(shape) == 0

    call_triton_wait_signal = f"helion.runtime.triton_wait_{'' if is_scalar else 'multiple_'}signal(addr={{bar_addrs}}, expect={{signal}}, update={{update}}, sem='{sem}', scope='{scope}', op='{op}', skip_sync={skip_sync})"

    return expr_from_string(
        call_triton_wait_signal,
        bar_addrs=bar_addrs_expr,
        signal=signal_expr,
        update=update_expr,
    )


@has_side_effect
@_decorators.api(tiles_as_sizes=True, allow_host_tensor=True)
def signal(
    signal_pad: torch.Tensor | StackTensor,
    index: list[object] | None = None,
    signal: int = 1,
    wait_for: int | None = None,
    scope: str = "gpu",
    hasPreviousMemAccess: bool = True,
) -> torch.Tensor:
    """
    Set global memory barriers.

    Sets global memory barriers to the specified value.
    If wait_for is not None, it waits for the barriers to be cleared before setting.

    Args:
        signal_pad: Tensor of global memory barriers to set
        index: Indices to index into the signal_pad tensor
        signal: the value to send
        wait_for: The value to wait for before sending the signal.
        scope: The scope of the lock (default: 'gpu')
        hasPreviousMemAccess: Whether the signal is preceded by a memory access (default: True)

    Returns:
        The old value of the global memory barriers before the update.
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(signal)
def _(
    signal_pad: torch.Tensor | StackTensor,
    index: list[object],
    signal: int = 1,
    wait_for: int | None = None,
    scope: str = "gpu",
    hasPreviousMemAccess: bool = True,
) -> tuple[torch.Tensor | tuple, object, int, int | None, str, bool]:
    from .tile_proxy import Tile

    assert isinstance(signal_pad, (torch.Tensor, StackTensor))

    if signal_pad.dtype not in (torch.int32, torch.uint32):
        raise NotImplementedError(
            f"Unsupported signal pad dtype: {signal_pad.dtype}. Must be of torch.int32 or torch.uint32."
        )

    valid_scopes = {"sys", "gpu"}

    if scope not in valid_scopes:
        raise ValueError(f"Invalid scope '{scope}'. Must be one of {valid_scopes}.")

    index = Tile._prepare_index(index)
    index = Tile._tiles_to_sizes(index)

    if isinstance(signal_pad, StackTensor):
        return (tuple(signal_pad), index, signal, wait_for, scope, hasPreviousMemAccess)
    return (signal_pad, index, signal, wait_for, scope, hasPreviousMemAccess)


@_decorators.register_fake(signal)
def _(
    signal_pad: torch.Tensor | tuple,
    index: list[object],
    signal: int = 1,
    wait_for: int | None = None,
    scope: str = "gpu",
    hasPreviousMemAccess: bool = True,
) -> torch.Tensor:
    if isinstance(signal_pad, tuple):
        signal_pad = StackTensor(*signal_pad)
        stack_shape = signal_pad.dev_ptrs.shape
        subscript_shape = SubscriptIndexing.compute_shape(signal_pad.tensor_like, index)
        shape = list(stack_shape) + subscript_shape
    elif isinstance(signal_pad, torch.Tensor):
        shape = SubscriptIndexing.compute_shape(signal_pad, index)
    else:
        raise NotImplementedError(f"Unsupported signal pad type: {type(signal_pad)}")

    return signal_pad.new_empty(shape)


@_decorators.codegen(signal)
def _(state: CodegenState) -> ast.AST:
    import ast

    from .._compiler.ast_extension import expr_from_string
    from .._compiler.indexing_strategy import SubscriptIndexing

    signal_pad = state.proxy_arg(0)
    index = state.proxy_arg(1)
    signal = state.proxy_arg(2)
    wait_for = state.proxy_arg(3)
    scope = state.proxy_arg(4)
    hasPreviousMemAccess = state.proxy_arg(5)

    if isinstance(signal_pad, tuple):
        signal_pad = StackTensor(*signal_pad)
    assert isinstance(signal_pad, (torch.Tensor, StackTensor))
    assert isinstance(index, list)

    assert type(scope) is str

    assert type(hasPreviousMemAccess) is bool

    sem = "release" if hasPreviousMemAccess else "relaxed"
    op = "atomic_xchg" if wait_for is None else "atomic_cas"
    skip_sync = not hasPreviousMemAccess

    if isinstance(signal_pad, torch.Tensor):
        indices = SubscriptIndexing.create(state, signal_pad, index)
        shape = SubscriptIndexing.compute_shape(signal_pad, index)
        signal_pad_name = state.device_function.tensor_arg(signal_pad).name

        bar_addrs_expr = expr_from_string(
            f"{signal_pad_name} + {{offset}}", offset=indices.index_expr
        )
    elif isinstance(signal_pad, StackTensor):
        from .._compiler.indexing_strategy import StackIndexingStrategy

        subscript_shape = SubscriptIndexing.compute_shape(signal_pad.tensor_like, index)
        stack_shape = signal_pad.dev_ptrs.shape
        shape = subscript_shape + list(stack_shape)

        stack_broadcast, tensor_broadcast = StackIndexingStrategy.get_broadcast_str(
            stack_shape, subscript_shape
        )

        tensor_like_indices = SubscriptIndexing.create(
            state, signal_pad.tensor_like, index
        )

        dtype = triton_type(signal_pad.dtype)

        ast_tensors = state.ast_args[0]
        assert isinstance(ast_tensors, tuple)
        assert len(ast_tensors) == 2
        tensor_like_ast, dev_ptrs_ast = ast_tensors
        bar_addrs_expr = expr_from_string(
            f"{{base}}.to(tl.pointer_type({dtype})){stack_broadcast} + {{offset}}{tensor_broadcast}",
            base=dev_ptrs_ast,
            offset=tensor_like_indices.index_expr,
        )
    else:
        raise NotImplementedError(f"Unsupported signal pad type: {type(signal_pad)}")

    is_scalar = len(shape) == 0

    signal_expr = ast.Constant(value=signal)  # pyright: ignore[reportArgumentType]
    if wait_for is not None:
        wait_for_expr = ast.Constant(value=wait_for)  # pyright: ignore[reportArgumentType]
    else:
        wait_for_expr = ast.Constant(value=0)
    skip_sync_expr = ast.Constant(value=skip_sync)  # pyright: ignore[reportArgumentType]

    if wait_for is not None:
        call_triton_wait_signal = f"helion.runtime.triton_wait_{'' if is_scalar else 'multiple_'}signal(addr={{bar_addrs}}, expect={{wait_for}}, update={{signal}}, sem='{sem}', scope='{scope}', op='{op}', skip_sync=True, sync_before=(not {{skip_sync}}))"
        return expr_from_string(
            call_triton_wait_signal,
            bar_addrs=bar_addrs_expr,
            wait_for=wait_for_expr,
            signal=signal_expr,
            skip_sync=skip_sync_expr,
        )
    return expr_from_string(
        f"helion.runtime.triton_send_signal(addr={{bar_addrs}}, update={{signal}}, sem='{sem}', scope='{scope}', op='{op}', skip_sync={{skip_sync}})",
        bar_addrs=bar_addrs_expr,
        signal=signal_expr,
        skip_sync=skip_sync_expr,
    )
