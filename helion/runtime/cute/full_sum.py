"""One host ``sum``/``mean`` of a 1-D FP32 tensor with ATen's single-CTA tree.

``torch.sum(x)`` / ``x.mean()`` of a contiguous 1-D FP32 tensor of at least
128 elements that ATen reduces in one CTA (``setReduceConfig``:
``vectorize_input``, ``block_width = min(last_pow2(numel // 4), 512)``, no
cross-CTA pass) is computed by :func:`full_sum_kernel`
(``helion._compiler.cute.full_sum_runtime``), which reproduces that addition
tree exactly. The same ATen fingerprint as the paired sum gates the
replacement; every unsupported case executes the original host call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

from .paired_sum import _compatible_aten_sum
from .paired_sum import _ordinary_tensor

if TYPE_CHECKING:
    from collections.abc import Callable

MIN_VECTORIZED_NUMEL = 128
MAX_BLOCK_WIDTH = 512
MAX_VALUES_PER_THREAD = 256
# ``alignof(aligned_vector<float, 4>)``: a less aligned input makes ATen
# reduce a head of elements separately (another tree).
INPUT_ALIGNMENT = 16


@dataclass(frozen=True)
class FullSumPlan:
    numel: int
    width: int
    factor: float  # ATen's MeanOps factor (float32(1) / numel), 1.0 for sum


def _last_pow2(value: int) -> int:
    return 1 << (value.bit_length() - 1)


def full_sum_plan(numel: int) -> FullSumPlan | None:
    """``None`` when ATen would use another tree (short inputs, global reduce)."""
    if numel < MIN_VECTORIZED_NUMEL:
        return None
    lanes = numel // 4
    width = _last_pow2(lanes) if lanes < MAX_BLOCK_WIDTH else MAX_BLOCK_WIDTH
    values_per_thread = -(-numel // width)
    if values_per_thread >= MAX_VALUES_PER_THREAD:
        return None
    # ``static_cast<float>(num_output_elements) / numel`` in FP32 arithmetic.
    factor = float(
        (
            torch.tensor(1.0, dtype=torch.float32)
            / torch.tensor(float(numel), dtype=torch.float32)
        ).item()
    )
    return FullSumPlan(numel, width, factor)


def try_full_sum(
    tensor: torch.Tensor,
    *,
    mean: bool,
    _launcher: Callable[..., object],
) -> torch.Tensor | None:
    """Return the sum/mean as a fresh 0-d tensor, or ``None`` to run the original.

    Unsupported dispatch, gradients, dtypes, layouts, alignments, lengths and
    library builds keep the exact original ``torch.sum`` / ``mean`` call.
    """
    if (
        torch._C._is_torch_function_mode_enabled()
        or _get_current_dispatch_mode_stack()
        or torch.Tensor.sum is not torch._C.TensorBase.sum
        or torch.Tensor.mean is not torch._C.TensorBase.mean
        or torch.sum is not torch._C._VariableFunctions.sum
        or torch.mean is not torch._C._VariableFunctions.mean
        or torch.Tensor.new_empty is not torch._C.TensorBase.new_empty
    ):
        return None
    if (
        not _ordinary_tensor(tensor)
        or tensor.device.type != "cuda"
        or tensor.ndim != 1
        or tensor.dtype != torch.float32
        or tensor.requires_grad
        or tensor.stride(0) != 1
        or tensor.data_ptr() % INPUT_ALIGNMENT
    ):
        return None
    plan = full_sum_plan(tensor.size(0))
    if plan is None or not _compatible_aten_sum():
        return None
    from ..._compiler.cute.full_sum_runtime import full_sum_kernel

    output = tensor.new_empty((1,))
    _launcher(
        full_sum_kernel,
        (1, 1, 1),
        tensor,
        output,
        plan.numel,
        plan.width,
        plan.factor if mean else 1.0,
        mean,
        block=(plan.width, 1, 1),
    )
    return output.view(())
