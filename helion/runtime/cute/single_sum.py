"""One returned sum/cast using the paired helper's audited FP32 addition tree.

The tree and byte-offset bounds are shared with paired_sum. Float16 additionally
requires the audited CUDA Half(float) constructor, which uses __float2half: the
CUDA intrinsic rounds to nearest, ties to even. The device kernel performs that
conversion only after the last FP32 addition. No NaN payload identity is assumed.
"""

from __future__ import annotations

from functools import cache
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

from .paired_sum import SumPlan
from .paired_sum import _compatible_aten_sum
from .paired_sum import _ordinary_tensor
from .paired_sum import sum_plan

if TYPE_CHECKING:
    from collections.abc import Callable

_ATEN_HALF_HEADER = "torch/headeronly/util/Half.h"
_ATEN_HALF_SHA256 = "6c02317b90c2d1265b7d8d29d3f1f39e28abf41b6d94afce4d91c297623a90c6"


@cache
def _compatible_aten_half_cast() -> bool:
    # The library build and sum implementation are checked by the shared gate.
    # Fingerprint this extra constructor once, not on repeated kernel calls.
    filename = Path(torch.__file__).parent / "include" / _ATEN_HALF_HEADER
    try:
        actual = hashlib.sha256(filename.read_bytes()).hexdigest()
    except OSError:
        return False
    return actual == _ATEN_HALF_SHA256


def plan_for_single(
    tensor: torch.Tensor, dtype: torch.dtype, layout: str
) -> SumPlan | None:
    if (
        tensor.ndim != 2
        or tensor.dtype != torch.float32
        or tensor.requires_grad
        or tensor.stride(1) != 1
        or dtype not in (torch.float16, torch.bfloat16, torch.float32)
    ):
        return None
    return sum_plan(
        tensor.size(0), tensor.size(1), tensor.stride(0), tensor.data_ptr(), layout
    )


def try_single_sum_cast(
    tensor: torch.Tensor,
    dtype_argument: torch.Tensor | torch.dtype,
    *,
    dtype_is_tensor: bool,
    layout: str,
    _launcher: Callable[..., object],
) -> torch.Tensor | None:
    """Allocate one fresh result or leave the original return to execute.

    All producer calls, input/partial allocations and other returned values
    remain in the original generated caller. Unsupported dispatch, gradients,
    layouts and implementation identities retain its exact sum and cast calls.
    The supplied production launcher resolves the current stream on every call.
    """
    if (
        torch._C._is_torch_function_mode_enabled()
        or _get_current_dispatch_mode_stack()
        or torch.Tensor.sum is not torch._C.TensorBase.sum
        or torch.Tensor.to is not torch._C.TensorBase.to
        or torch.Tensor.new_empty is not torch._C.TensorBase.new_empty
    ):
        return None
    if not _ordinary_tensor(tensor) or tensor.device.type != "cuda":
        return None
    if dtype_is_tensor:
        if not _ordinary_tensor(dtype_argument):
            return None
        assert isinstance(dtype_argument, torch.Tensor)
        dtype = dtype_argument.dtype
    else:
        if not isinstance(dtype_argument, torch.dtype):
            return None
        dtype = dtype_argument
    plan = plan_for_single(tensor, dtype, layout)
    if plan is None or not _compatible_aten_sum():
        return None
    if dtype == torch.float16 and not _compatible_aten_half_cast():
        return None
    from ..._compiler.cute.single_sum_runtime import single_sum_cast_kernel

    output = tensor.new_empty((plan.columns,), dtype=dtype)
    if plan.columns:
        _launcher(
            single_sum_cast_kernel,
            plan.grid,
            tensor,
            output,
            plan.rows,
            plan.columns,
            plan.row_lanes,
            plan.column_threads,
            plan.vector,
            block=plan.block,
        )
    return output
