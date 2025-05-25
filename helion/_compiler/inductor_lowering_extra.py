"""
NOTE: This file is only designed to be imported within `patch_inductor_lowerings()` function.
"""

from __future__ import annotations

import torch
from torch._inductor.lowering import register_lowering as register_inductor_lowering


def var_mean_helper_(
    x: torch._inductor.ir.TensorBox,
    *,
    axis: list[int] | None,
    correction: float | None,
    keepdim: bool,
    return_mean: bool,
) -> torch._inductor.ir.IRNode:
    from torch._inductor.lowering import to_dtype
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
    output = tuple(to_dtype(x, out_dtype, copy=False) for x in output)
    return output[0] if not return_mean else output


@register_inductor_lowering(torch.ops.aten.var_mean.correction)  # pyre-ignore[56]
def var_mean(
    x: torch._inductor.ir.TensorBox,
    axis: list[int] | None = None,
    *,
    correction: float | None = None,
    keepdim: bool = False,
) -> torch._inductor.ir.IRNode:
    return var_mean_helper_(
        x, axis=axis, correction=correction, keepdim=keepdim, return_mean=True
    )
