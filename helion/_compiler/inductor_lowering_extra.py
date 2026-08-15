from __future__ import annotations

import contextlib
import functools
import threading
from typing import Any
from typing import Callable
from typing import Generator

import torch
from torch._inductor.ir import TensorBox
from torch._inductor.lowering import to_dtype

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
