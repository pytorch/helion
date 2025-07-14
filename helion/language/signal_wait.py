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

__all__ = ["signal", "wait"]


@has_side_effect
@_decorators.api(tiles_as_sizes=True, allow_host_tensor=True)
def wait(
    signal_pad: torch.Tensor,
    index: list[object],
    signal: int = 1,
    update: int | None = None,
    op: str = "ld",
    sem: str = "acquire",
    scope: str = "gpu",
    skip_sync: bool = False,
) -> None:
    """Wait until all entries of the signal_pad slice are equal to the signal value.
    Args:
        signal_pad: The signal pad tensor to wait on
        index: Indices to index into the signal_pad tensor
        signal: the value to wait for
        update: Atomically update the signal_pad tensor with this value once the signal is observed. (default: None)
        op: The memory op for acquiring the lock (default: 'ld')
        sem: The memory semantic for acquiring the lock (default: 'acquire')
        scope: The scope of the lock (default: 'gpu')
        skip_sync: Skip the syncthreads after the wait (default: False)

    Returns:
        None
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(wait)
def _(
    signal_pad: torch.Tensor,
    index: list[object],
    signal: int = 1,
    update: int | None = None,
    op: str = "ld",
    sem: str = "acquire",
    scope: str = "gpu",
    skip_sync: bool = False,
) -> tuple[torch.Tensor, object, int, int | None, str, str, str, bool]:
    from .tile_proxy import Tile

    valid_ops = {"ld", "atomic_cas"}
    valid_sems = {"relaxed", "acquire", "acq_rel"}
    valid_scopes = {"sys", "gpu"}

    if op not in valid_ops:
        raise ValueError(f"Invalid Wait op '{op}'. Must be one of {valid_ops}. ")

    if sem == "release":
        raise ValueError(
            f"Do not use '{sem}' for wait patterns. Wait sem must be one of {valid_sems}."
        )

    if sem not in valid_sems:
        raise ValueError(
            f"Invalid memory semantic '{sem}'. Must be one of {valid_sems}."
        )

    if op == "atomic_cas" and update is None:
        raise ValueError(
            f"{op} without an update value. Do you want to use 'ld' instead? "
        )

    if op == "ld":
        assert update is None
        update = 0

    if scope not in valid_scopes:
        raise ValueError(f"Invalid scope '{scope}'. Must be one of {valid_scopes}.")

    index = Tile._prepare_index(index)
    index = Tile._tiles_to_sizes(index)

    return (signal_pad, index, signal, update, op, sem, scope, skip_sync)


@_decorators.register_fake(wait)
def _(
    signal_pad: torch.Tensor,
    index: list[object],
    signal: int = 1,
    update: int | None = None,
    op: str = "ld",
    sem: str = "acquire",
    scope: str = "sys",
    skip_sync: bool = False,
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
    op = state.proxy_arg(4)
    sem = state.proxy_arg(5)
    scope = state.proxy_arg(6)
    skip_sync = state.proxy_arg(7)

    assert isinstance(signal_pad, torch.Tensor)
    assert isinstance(index, (list))

    indices = SubscriptIndexing.create(state, signal_pad, index)
    signal_pad_name = state.device_function.tensor_arg(signal_pad).name

    signal_expr = ast.Constant(value=signal)  # pyright: ignore[reportArgumentType]
    update_expr = ast.Constant(value=update)  # pyright: ignore[reportArgumentType]

    assert type(op) is str
    assert type(sem) is str
    assert type(scope) is str

    bar_tensor_shape = SubscriptIndexing.compute_shape(signal_pad, index)
    is_scalar = len(bar_tensor_shape) == 0

    if is_scalar:
        call_triton_wait_signal = f"helion.runtime.triton_wait_signal(addr={signal_pad_name} + offset, expect=signal, update=update, sem='{sem}', scope='{scope}', op='{op}', skip_sync={skip_sync})"
    else:
        if signal_pad.dtype not in (torch.int32, torch.uint32):
            raise NotImplementedError(
                f"Unsupported signal pad dtype: {signal_pad.dtype}. Must be of torch.int32 or torch.uint32."
            )
        call_triton_wait_signal = f"helion.runtime.triton_wait_multiple_signal(addr={signal_pad_name} + offset, expect=signal, update=update, sem='{sem}', scope='{scope}', op='{op}', skip_sync={skip_sync})"

    return expr_from_string(
        call_triton_wait_signal,
        offset=indices.index_expr,
        signal=signal_expr,
        update=update_expr,
    )


@has_side_effect
@_decorators.api(tiles_as_sizes=True, allow_host_tensor=True)
def signal(
    signal_pad: torch.Tensor,
    index: list[object],
    signal: int = 1,
    wait_for: int | None = None,
    op: str = "atomic_xchg",
    sem: str = "release",
    scope: str = "gpu",
    skip_sync: bool = False,
) -> torch.Tensor:
    """Set the signal_pad slice to the signal value.
    Args:
        signal_pad: The signal pad to signal
        index: Indices to index into the signal_pad tensor
        signal: the value to send
        wait_for: The value to wait for before sending the signal. Only valid for op = 'atomic_cas'.
        op: The memory op for acquiring the lock (default: 'atomic_xchg')
        sem: The memory semantic for acquiring the lock (default: 'release')
        scope: The scope of the lock (default: 'gpu')
        skip_sync: Skip the syncthreads before sending signal (default: False)
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(signal)
def _(
    signal_pad: torch.Tensor,
    index: list[object],
    signal: int = 1,
    wait_for: int | None = None,
    op: str = "atomic_xchg",
    sem: str = "release",
    scope: str = "gpu",
    skip_sync: bool = False,
) -> tuple[torch.Tensor, object, int, int | None, str, str, str, bool]:
    from .tile_proxy import Tile

    valid_ops = {"atomic_add", "atomic_xchg", "atomic_cas"}
    valid_sems = {"relaxed", "release", "acq_rel"}
    valid_scopes = {"sys", "gpu"}

    if op not in valid_ops:
        raise ValueError(f"Invalid signal op '{op}'. Must be one of {valid_ops}. ")

    if op == "atomic_cas" and wait_for is None:
        raise ValueError(
            f"{op} without a wait_for value. Do you want to use 'atomic_add' or 'atomic_xchg' instead? "
        )
    if op in {"atomic_add", "atomic_xchg"} and wait_for is not None:
        raise ValueError(
            f"{op} with a wait_for value. Do you want to use 'atomic_cas' instead? "
        )

    if sem not in valid_sems:
        raise ValueError(
            f"Invalid memory semantic '{sem}'. Must be one of {valid_sems}."
        )

    if scope not in valid_scopes:
        raise ValueError(f"Invalid scope '{scope}'. Must be one of {valid_scopes}.")

    index = Tile._prepare_index(index)
    index = Tile._tiles_to_sizes(index)

    return (signal_pad, index, signal, wait_for, op, sem, scope, skip_sync)


@_decorators.register_fake(signal)
def _(
    signal_pad: torch.Tensor,
    index: list[object],
    signal: int = 1,
    wait_for: int | None = None,
    op: str = "atomic_xchg",
    sem: str = "release",
    scope: str = "gpu",
    skip_sync: bool = False,
) -> torch.Tensor:
    return signal_pad.new_empty(SubscriptIndexing.compute_shape(signal_pad, index))


@_decorators.codegen(signal)
def _(state: CodegenState) -> ast.AST:
    import ast

    from .._compiler.ast_extension import expr_from_string
    from .._compiler.indexing_strategy import SubscriptIndexing

    signal_pad = state.proxy_arg(0)
    index = state.proxy_arg(1)
    signal = state.proxy_arg(2)
    wait_for = state.proxy_arg(3)
    op = state.proxy_arg(4)
    sem = state.proxy_arg(5)
    scope = state.proxy_arg(6)
    skip_sync = state.proxy_arg(7)

    assert isinstance(signal_pad, torch.Tensor)
    assert isinstance(index, list)

    indices = SubscriptIndexing.create(state, signal_pad, index)
    signal_pad_name = state.device_function.tensor_arg(signal_pad).name

    signal_expr = ast.Constant(value=signal)  # pyright: ignore[reportArgumentType]
    if wait_for is not None:
        wait_for_expr = ast.Constant(value=wait_for)  # pyright: ignore[reportArgumentType]
    else:
        wait_for_expr = ast.Constant(value=0)
    skip_sync_expr = ast.Constant(value=skip_sync)  # pyright: ignore[reportArgumentType]
    assert type(op) is str
    assert type(sem) is str
    assert type(scope) is str

    if op == "atomic_cas":
        bar_tensor_shape = SubscriptIndexing.compute_shape(signal_pad, index)
        is_scalar = len(bar_tensor_shape) == 0
        if is_scalar:
            call_triton_wait_signal = f"helion.runtime.triton_wait_signal(addr={signal_pad_name} + offset, expect=wait_for, update=signal, sem='{sem}', scope='{scope}', op='{op}', skip_sync=True, sync_before=(not skip_sync))"
        else:
            call_triton_wait_signal = f"helion.runtime.triton_wait_multiple_signal(addr={signal_pad_name} + offset, expect=wait_for, update=signal, sem='{sem}', scope='{scope}', op='{op}', skip_sync=True, sync_before=(not skip_sync))"

        return expr_from_string(
            call_triton_wait_signal,
            offset=indices.index_expr,
            wait_for=wait_for_expr,
            signal=signal_expr,
            skip_sync=skip_sync_expr,
        )
    call_triton_send_signal = f"helion.runtime.triton_send_signal(addr={signal_pad_name} + offset, update=signal, sem='{sem}', scope='{scope}', op='{op}', skip_sync=skip_sync)"

    return expr_from_string(
        call_triton_send_signal,
        offset=indices.index_expr,
        signal=signal_expr,
        skip_sync=skip_sync_expr,
    )
