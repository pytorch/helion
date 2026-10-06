from __future__ import annotations

import contextlib
import functools
import threading
from typing import Any
from typing import Callable
from typing import Generator

import sympy
import torch
from torch._inductor import ir
from torch._inductor.ir import TensorBox
from torch._inductor.lowering import to_dtype

from .. import exc

inductor_lowering_dispatch: dict[Callable[..., Any] | str, Callable[..., Any]] = {}

_MISSING_LOWERING = object()
_patch_lock = threading.Lock()
_patch_users = 0
_patch_table: dict[Any, Any] | None = None
_patch_entries: dict[Any, tuple[object, object]] = {}


def create_fp16_to_fp32_unary_fallback_lowering(
    original_op: Callable[..., object],
) -> Callable[..., object]:
    """Create a lowering that converts fp16/bfloat16 inputs to fp32 before calling the operation."""

    @functools.wraps(original_op)
    def fp32_fallback_lowering(x: object) -> object:
        from .compile_environment import CompileEnvironment

        if (
            not CompileEnvironment.has_current()
            or CompileEnvironment.current().backend_name == "pallas"
        ):
            return original_op(x)
        if isinstance(x, TensorBox) and (original_dtype := x.get_dtype()) in (
            torch.float16,
            torch.bfloat16,
        ):
            x_fp32 = to_dtype(x, torch.float32)
            result_fp32 = original_op(x_fp32)
            assert isinstance(result_fp32, TensorBox)
            return to_dtype(result_fp32, original_dtype)
        return original_op(x)

    return fp32_fallback_lowering


def _compile_environment_lowering(
    op: Callable[..., Any] | str,
    patched: Callable[..., Any],
    previous: object,
) -> Callable[..., Any]:
    """Use a Helion override only in the thread compiling a Helion kernel."""

    @functools.wraps(patched)
    def scoped(*args: object, **kwargs: object) -> object:
        from .compile_environment import CompileEnvironment

        if CompileEnvironment.has_current():
            return patched(*args, **kwargs)
        if previous is _MISSING_LOWERING:
            raise KeyError(f"no Inductor lowering registered for {op!r}")
        return previous(*args, **kwargs)  # pyrefly: ignore [not-callable]

    return scoped


def _restore_inductor_lowerings() -> None:
    """Restore Helion-owned entries without disturbing concurrent registrations."""
    global _patch_table

    assert _patch_table is not None
    for op, (previous, installed) in _patch_entries.items():
        if _patch_table.get(op, _MISSING_LOWERING) is not installed:
            continue
        if previous is _MISSING_LOWERING:
            _patch_table.pop(op, None)
        else:
            _patch_table[op] = previous
    _patch_entries.clear()
    _patch_table = None


# Operations that need fp32 fallbacks due to libdevice/tl_math limitations
FP32_FALLBACK_OPS_UNARY = [
    torch.ops.aten.rsqrt.default,
    torch.ops.aten.sqrt.default,
    torch.ops.aten.sin.default,
    torch.ops.aten.cos.default,
    torch.ops.aten.log.default,
    torch.ops.aten.tanh.default,
    torch.ops.aten.log1p.default,
    torch.ops.aten.expm1.default,
    torch.ops.aten.exp.default,
]


@contextlib.contextmanager
def patch_inductor_lowerings() -> Generator[None, Any, Any]:
    """Temporarily install lowering overrides needed by Helion compilation.

    Inductor's lowering table is process-global, so the installed wrappers
    apply Helion behavior only with an active compile environment and delegate
    to the prior lowerings in all other threads.
    """
    global _patch_table, _patch_users

    with _patch_lock:
        if _patch_users == 0:
            # Mutate the existing table: register_lowering() captures this dict
            # object, and replacing it disconnects later registrations.
            # pyrefly: ignore [implicit-import]
            _patch_table = torch._inductor.lowering.lowerings
            try:
                for op, patched in inductor_lowering_dispatch.items():
                    previous = _patch_table.get(op, _MISSING_LOWERING)
                    installed = _compile_environment_lowering(op, patched, previous)
                    _patch_entries[op] = (previous, installed)
                    _patch_table[op] = installed
                for op in FP32_FALLBACK_OPS_UNARY:
                    current = _patch_table.get(op, _MISSING_LOWERING)
                    if current is _MISSING_LOWERING or not callable(current):
                        raise KeyError(f"no Inductor lowering registered for {op!r}")
                    existing = _patch_entries.get(op)
                    previous = current if existing is None else existing[0]
                    installed = create_fp16_to_fp32_unary_fallback_lowering(current)
                    _patch_entries[op] = (previous, installed)
                    _patch_table[op] = installed
            except Exception:
                _restore_inductor_lowerings()
                raise
        _patch_users += 1
    try:
        yield
    finally:
        with _patch_lock:
            _patch_users -= 1
            if _patch_users == 0:
                _restore_inductor_lowerings()


# pyrefly: ignore [implicit-import]
register_inductor_lowering = torch._inductor.lowering.register_lowering


def var_mean_helper_(
    # pyrefly: ignore [implicit-import]
    x: torch._inductor.ir.TensorBox,
    *,
    axis: list[int] | None,
    correction: float | None,
    keepdim: bool,
    return_mean: bool,
    # pyrefly: ignore [implicit-import]
) -> torch._inductor.ir.TensorBox:
    from torch._inductor.lowering import var_mean_sum_
    from torch._prims_common import get_computation_dtype

    out_dtype = x.get_dtype()
    compute_dtype = get_computation_dtype(out_dtype)

    x = to_dtype(x, compute_dtype, copy=False)

    kwargs = {
        "x": x,
        "axis": axis,
        "correction": correction,
        "keepdim": keepdim,
        "return_mean": return_mean,
    }
    # TODO(yf225): support Welford reduction in Helion, then switch back to use Inductor `var_mean_helper_()`.
    output = var_mean_sum_(**kwargs)
    output = tuple(to_dtype(o, out_dtype, copy=False) for o in output)
    # pyrefly: ignore [bad-return]
    return output[0] if not return_mean else output


_JAGGED_MEAN_UNSUPPORTED = (
    "a mean over the jagged tile dim is not supported: each row of the "
    "parent tile holds its own number of elements, so there is no one "
    "count to divide by; divide the sum by the row's length instead"
)


def _reduced_dim_extent(size: sympy.Expr) -> sympy.Expr:
    """The elements a mean over a dim of ``size`` divides by.

    A dim sized by the block of an ``hl.tile`` loop may hold fewer elements
    than the block: a block wider than the dim, or the last tile of a dim
    the block does not divide.  The masked elements are already out of the
    sum, so the mean divides by the tile's extent (``tile.end -
    tile.begin``, rendered as the block size where no mask is needed; see
    :class:`~helion._compiler.variable_origin.TileExtentOrigin`) instead of
    the block.  A reduction dim is its full size and is not changed here.
    A jagged tile dim has a per-row extent (each row of the parent tile
    holds its own number of elements) that no scalar divisor expresses, so
    a mean over it is rejected rather than divided by the block, whether the
    size is the block itself or derived from it (``torch.cat([v, v], dim=1)``
    along the jagged dim sizes its dim ``2 * block``).  The tile cannot be
    flattened into a sibling loop afterwards, as for ``tile.end``.
    """
    from ..language.tile_ops import _disable_flatten_get_tile
    from .compile_environment import CompileEnvironment
    from .host_function import HostFunction
    from .host_function import SymbolOrigin
    from .variable_origin import TileExtentOrigin

    env = CompileEnvironment.current()
    if not isinstance(size, sympy.Symbol):
        if any(
            (block_id := env.get_block_id(symbol)) is not None
            and env.is_jagged_tile(block_id)
            for symbol in size.free_symbols
        ):
            raise exc.InvalidJaggedTileUsage(_JAGGED_MEAN_UNSUPPORTED)
        return size
    block_id = env.get_block_id(size)
    if block_id is None:
        return size
    info = env.block_sizes[block_id]
    if info.reduction or info.var._sympy_() != size:  # pyrefly: ignore [missing-attribute]
        return size
    if env.is_jagged_tile(block_id):
        raise exc.InvalidJaggedTileUsage(_JAGGED_MEAN_UNSUPPORTED)
    extent = env.cached_create_unbacked_symint(("tile_extent", info.var))._sympy_()
    assert isinstance(extent, sympy.Symbol)
    HostFunction.current().expr_to_origin[extent] = SymbolOrigin(
        TileExtentOrigin(block_id)
    )
    _disable_flatten_get_tile(info.var)
    return extent


@register_inductor_lowering(
    # The overloads Helion traces, spelled out: a packet registers only
    # itself when inductor's own table already lists its overloads.
    [torch.ops.aten.mean.dim, torch.ops.aten.mean.default],
    lowering_dict=inductor_lowering_dispatch,
)
def mean(
    x: TensorBox,
    axis: list[int] | int | None = None,
    keepdim: bool = False,
    *,
    dtype: torch.dtype | None = None,
) -> TensorBox:
    """Inductor's ``mean`` with the divisor of a tile dim being the tile's extent."""
    from torch._inductor.lowering import _validate_reduction_axis
    from torch._inductor.lowering import div
    from torch._inductor.lowering import sum_

    if dtype is not None:
        x = to_dtype(x, dtype)
    size = x.get_size()
    axis = _validate_reduction_axis(x, axis)
    # Computed in higher precision until the end of the lowering, as inductor does.
    output_dtype = x.get_dtype()
    if output_dtype in (torch.float16, torch.bfloat16):
        x = to_dtype(x, torch.float)
    sum_result = sum_(x, axis, keepdim)
    denom = sympy.Mul(*[_reduced_dim_extent(size[i]) for i in axis])
    device = x.get_device()
    assert device is not None
    denom_box = ir.IndexingConstant(index=denom, dtype=x.get_dtype(), device=device)
    expanded = ir.ExpandView.create(denom_box, list(sum_result.get_size()))
    return to_dtype(div(sum_result, expanded), output_dtype)


@register_inductor_lowering(
    [torch.ops.aten.var.correction],
    lowering_dict=inductor_lowering_dispatch,
)
def var_(
    # pyrefly: ignore [implicit-import]
    x: torch._inductor.ir.TensorBox,
    axis: list[int] | None = None,
    *,
    correction: float | None = None,
    keepdim: bool = False,
    # pyrefly: ignore [implicit-import]
) -> torch._inductor.ir.TensorBox:
    return var_mean_helper_(
        x,
        axis=axis,
        correction=correction,
        keepdim=keepdim,
        return_mean=False,
    )


@register_inductor_lowering(
    torch.ops.aten.var_mean.correction,
    lowering_dict=inductor_lowering_dispatch,
)
def var_mean(
    # pyrefly: ignore [implicit-import]
    x: torch._inductor.ir.TensorBox,
    axis: list[int] | None = None,
    *,
    correction: float | None = None,
    keepdim: bool = False,
    # pyrefly: ignore [implicit-import]
) -> torch._inductor.ir.TensorBox:
    return var_mean_helper_(
        x,
        axis=axis,
        correction=correction,
        keepdim=keepdim,
        return_mean=True,
    )
