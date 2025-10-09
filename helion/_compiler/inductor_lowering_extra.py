from __future__ import annotations

import contextlib
import functools
from typing import Any
from typing import Callable
from typing import Generator

import torch
from torch._inductor.ir import TensorBox
from torch._inductor.lowering import lowerings as original_lowerings
from torch._inductor.lowering import to_dtype

inductor_lowering_dispatch: dict[Callable[..., Any] | str, Callable[..., Any]] = {}


def create_fp16_to_fp32_unary_fallback_lowering(
    original_op: Callable[..., object],
) -> Callable[..., object]:
    """Create a lowering that converts fp16/bfloat16 inputs to fp32 before calling the operation."""

    @functools.wraps(original_op)
    def fp32_fallback_lowering(x: object) -> object:
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


# Operations that need fp32 fallbacks due to libdevice/tl_math limitations
FP32_FALLBACK_OPS_UNARY = [
    torch.ops.aten.rsqrt.default,  # pyright: ignore[reportAttributeAccessIssue]
    torch.ops.aten.sqrt.default,  # pyright: ignore[reportAttributeAccessIssue]
    torch.ops.aten.sin.default,  # pyright: ignore[reportAttributeAccessIssue]
    torch.ops.aten.cos.default,  # pyright: ignore[reportAttributeAccessIssue]
    torch.ops.aten.log.default,  # pyright: ignore[reportAttributeAccessIssue]
    torch.ops.aten.tanh.default,  # pyright: ignore[reportAttributeAccessIssue]
    torch.ops.aten.log1p.default,  # pyright: ignore[reportAttributeAccessIssue]
    torch.ops.aten.expm1.default,  # pyright: ignore[reportAttributeAccessIssue]
    torch.ops.aten.exp.default,  # pyright: ignore[reportAttributeAccessIssue]
]

# Register fp32 fallback lowerings for ops that don't support fp16/bfloat16
for op in FP32_FALLBACK_OPS_UNARY:
    inductor_lowering_dispatch[op] = create_fp16_to_fp32_unary_fallback_lowering(
        original_lowerings[op]
    )


@contextlib.contextmanager
def patch_inductor_lowerings() -> Generator[None, Any, Any]:
    """Context manager to temporarily patch the inductor lowering table.

    This is useful for overwriting specific Inductor lowerings without
    affecting the global state, especially in cases where Helion
    is missing support for a specific lowering.
    """
    original_lowerings = torch._inductor.lowering.lowerings.copy()  # pyright: ignore[reportAttributeAccessIssue]
    try:
        torch._inductor.lowering.lowerings.update(inductor_lowering_dispatch)  # pyright: ignore[reportAttributeAccessIssue]
        yield
    finally:
        torch._inductor.lowering.lowerings = original_lowerings  # pyright: ignore[reportAttributeAccessIssue]


register_inductor_lowering = torch._inductor.lowering.register_lowering  # pyright: ignore[reportAttributeAccessIssue]


def var_mean_helper_(
    x: torch._inductor.ir.TensorBox,  # pyright: ignore[reportAttributeAccessIssue]
    *,
    axis: list[int] | None,
    correction: float | None,
    keepdim: bool,
    return_mean: bool,
) -> torch._inductor.ir.TensorBox:  # pyright: ignore[reportAttributeAccessIssue]
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
    return output[0] if not return_mean else output


@register_inductor_lowering(
    [torch.ops.aten.var.correction],  # pyright: ignore[reportAttributeAccessIssue]
    lowering_dict=inductor_lowering_dispatch,
)
def var_(
    x: torch._inductor.ir.TensorBox,  # pyright: ignore[reportAttributeAccessIssue]
    axis: list[int] | None = None,
    *,
    correction: float | None = None,
    keepdim: bool = False,
) -> torch._inductor.ir.TensorBox:  # pyright: ignore[reportAttributeAccessIssue]
    return var_mean_helper_(
        x,
        axis=axis,
        correction=correction,
        keepdim=keepdim,
        return_mean=False,
    )


@register_inductor_lowering(
    torch.ops.aten.var_mean.correction,  # pyright: ignore[reportAttributeAccessIssue]
    lowering_dict=inductor_lowering_dispatch,
)
def var_mean(
    x: torch._inductor.ir.TensorBox,  # pyright: ignore[reportAttributeAccessIssue]
    axis: list[int] | None = None,
    *,
    correction: float | None = None,
    keepdim: bool = False,
) -> torch._inductor.ir.TensorBox:  # pyright: ignore[reportAttributeAccessIssue]
    return var_mean_helper_(
        x,
        axis=axis,
        correction=correction,
        keepdim=keepdim,
        return_mean=True,
    )
