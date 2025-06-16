from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.fx import has_side_effect

from .. import exc
from . import _decorators

if TYPE_CHECKING:
    import ast

    from .._compiler.inductor_lowering import CodegenState


@has_side_effect
@_decorators.api(tiles_as_sizes=True)
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
        op: The memory op for acquring the lock (default: 'ld')
        sem: The memory sematic for acquring the lock (default: 'acquire')
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
    from helion.language.tile_proxy import Tile

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

    if op == "atomic_cas" and not update:
        raise ValueError(
            f"{op} without an update value. Do you want to use 'ld' instead? "
        )

    if op == "ld":
        assert update is None
        update = 0

    if scope not in valid_scopes:
        raise ValueError(f"Invalid scope '{scope}'. Must be one of {valid_scopes}.")

    # TODO(joydddd): add support for non scalar index into signal_pad
    for i in index:
        assert isinstance(i, int | torch.SymInt)

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

    signal_expr = ast.Constant(value=signal)
    update_expr = ast.Constant(value=update)

    assert type(op) is str
    assert type(sem) is str
    assert type(scope) is str

    call_triton_wait_signal = f"helion.runtime.triton_wait_signal(addr={signal_pad_name} + offset, expect=signal, update=update, sem='{sem}', scope='{scope}', op='{op}', skip_sync={skip_sync})"

    return expr_from_string(
        call_triton_wait_signal,
        offset=indices.index_expr,
        signal=signal_expr,
        update=update_expr,
    )
