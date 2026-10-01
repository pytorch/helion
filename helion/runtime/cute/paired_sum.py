"""Paired CUDA sum/cast with an explicit FP32 ordering compatibility contract.

ATen's public sum API does not specify a floating-point addition tree. We retain
its audited single-CTA, output-vectorized tree, rather than treating an arbitrary
reassociation as exact: four independent +0 FP32 chains, increasing chain fold,
then a descending power-of-two row tree, and one final cast per output. Mapped
and narrow differ only in the ownership of independent output columns.

The implementation fingerprint below deliberately falls back after an unaudited
Torch change. Updating it requires checking ReduceConfig, TensorIterator's byte
index splitting, sum's identity/accumulator type, and the reduction tree. Input
vectorization, byte-split and cross-CTA accumulation domains are not admitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

if TYPE_CHECKING:
    from collections.abc import Callable

# ATen sum dispatch plus the headers that define the admitted FP32 tree. This
# is a library-implementation compatibility gate, independent of workload or
# GPU shape. Missing sources or a different build retain ordinary Torch calls.
_ATEN_SUM_GIT = "332a69317e22b105a867838624a87984e05021e2"
_ATEN_SUM_HEADERS = {
    "ATen/native/cuda/thread_constants.h": "61e0f733a01a729f96036cd84b78645731d126534f3ecc972a3e91d3c37416d2",
    "ATen/native/cuda/Reduce.cuh": "8f04b32331911bd301938abe566ff6b3a390bd409ec3a98e1222d7ed698de09e",
    "ATen/native/ReduceOpsUtils.h": "3f4a5e3f51dc00b57c8ce86961ad50af1da48314efe340c8c57c59970b3d0b02",
    "ATen/native/SharedReduceOps.h": "75a6bad9a3c7f47572ab8264fd66487697e2a77bb9fba1b0d555d967aac18b5e",
}


@cache
def _compatible_aten_sum() -> bool:
    if (
        torch.version.git_version != _ATEN_SUM_GIT
        or torch.version.hip is not None
        or torch.version.cuda is None
    ):
        return False
    include = Path(torch.__file__).parent / "include"
    for relative, expected in _ATEN_SUM_HEADERS.items():
        try:
            actual = hashlib.sha256((include / relative).read_bytes()).hexdigest()
        except OSError:
            return False
        if actual != expected:
            return False
    return True


@dataclass(frozen=True)
class SumPlan:
    rows: int
    columns: int
    vector: int
    row_lanes: int
    column_threads: int
    reference_block: tuple[int, int]
    reference_splits_rows: bool
    layout: str

    @property
    def grid(self) -> tuple[int, int, int]:
        span = self.column_threads * self.vector
        return ((self.columns + span - 1) // span, 1, 1)

    @property
    def block(self) -> tuple[int, int, int]:
        return (self.column_threads, self.row_lanes, 1)


def _last_power_of_two(value: int) -> int:
    return 1 << (value.bit_length() - 1)


def sum_plan(
    rows: int,
    columns: int,
    row_stride: int,
    pointer_alignment: int,
    layout: str,
) -> SumPlan | None:
    """Mirror the non-global, output-vectorized Float32 CUDA sum tree.

    This tree is part of the explicit, fingerprinted ATen compatibility
    contract below. The choice changes column ownership only; it never changes
    which rows feed an FP32 accumulator, the fold order, or row-tree edges.
    """
    assert layout in ("mapped", "narrow")
    if (
        rows < 0
        or columns < 0
        or not columns <= row_stride < 2**31
        or pointer_alignment % 4
    ):
        return None
    if columns == 0:
        return SumPlan(rows, columns, 1, 1, 32, (32, 1), False, layout)
    # TensorIterator checks byte offsets before choosing CUDA's reduction tree.
    # A larger view is split into separate reductions and output accumulation;
    # even if our element offsets fit Int32, that can change FP32 rounding.
    last_element_offset = max(0, rows - 1) * row_stride + columns - 1
    if last_element_offset * 4 >= 2**31:
        return None
    vector = 4
    while pointer_alignment % (vector * 4) or columns % vector or row_stride % vector:
        vector //= 2
    if rows == 0:
        return SumPlan(rows, columns, vector, 1, 32, (32, 1), False, layout)
    # Real-storage TensorIterator coalesces the singleton output dimension.
    # CUDA then reduces across x (and may vectorize input), rather than using
    # this output-vectorized y tree. Meta tensors do not expose that coalescing.
    # Empty and one-row sums have no nontrivial addition tree to preserve.
    if rows > 1 and columns == 1:
        return None
    max_threads = 512 // vector
    dim0 = _last_power_of_two(min(columns // vector, max_threads))
    dim1 = _last_power_of_two(min(rows, max_threads))
    width = min(dim0, 32)
    height = min(dim1, max_threads // width)
    width = min(dim0, max_threads // height)
    split_rows = rows >= min(height * 16, 256)
    row_lanes = height if split_rows else 1
    column_threads = width if split_rows else width * height
    # Larger reductions may use CUDA's cross-CTA accumulation protocol. Keep
    # that unchanged instead of making an exact-tree claim for a different tree.
    if split_rows and (rows + row_lanes - 1) // row_lanes >= 256:
        return None
    if layout == "narrow":
        # One quarter as many independent columns, with at least one full warp
        # where the original launch already has that many threads. Row ownership
        # and every floating-point combine are unchanged.
        column_threads = min(
            column_threads,
            max(column_threads // 4, 32 // row_lanes, 1),
        )
    return SumPlan(
        rows,
        columns,
        vector,
        row_lanes,
        column_threads,
        (width, height),
        split_rows,
        layout,
    )


def plan_for_pair(
    left: torch.Tensor, right: torch.Tensor, dtype: torch.dtype, layout: str
) -> SumPlan | None:
    if (
        left.ndim != 2
        or right.ndim != 2
        or left.shape != right.shape
        or left.dtype != torch.float32
        or right.dtype != torch.float32
        or left.device != right.device
        or left.requires_grad
        or right.requires_grad
        or left.stride(1) != 1
        or right.stride(1) != 1
        or dtype not in (torch.bfloat16, torch.float32)
    ):
        return None
    first = sum_plan(
        left.size(0), left.size(1), left.stride(0), left.data_ptr(), layout
    )
    second = sum_plan(
        right.size(0), right.size(1), right.stride(0), right.data_ptr(), layout
    )
    # Input alignment can change ATen's vector width and hence its row tree.
    return first if first == second else None


def _ordinary_tensor(value: object) -> bool:
    return (
        type(value) is torch.Tensor
        and value.layout == torch.strided
        and not value.is_neg()
        and not value.is_conj()
        and not torch._is_functional_tensor(value)
        and not torch._C._functorch.is_functorch_wrapped_tensor(value)
        and torch.autograd.forward_ad.unpack_dual(value).tangent is None
    )


def try_paired_sum_cast(
    left: torch.Tensor,
    right: torch.Tensor,
    dtype_argument: torch.Tensor | torch.dtype,
    *,
    dtype_is_tensor: bool,
    layout: str,
    _launcher: Callable[..., object],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return None without tensor operations when original statements must run.

    The compiler retains the original sum and cast ASTs as the fallback. Their
    evaluation order, autograd and custom dispatch behavior therefore survive
    an unsupported runtime binding. Both successful outputs are fresh, in the
    original left/right order; all producer and partial allocations stay in the
    generated caller. The supplied launcher samples the current stream per call.
    """
    if (
        torch._C._is_torch_function_mode_enabled()
        or _get_current_dispatch_mode_stack()
        or torch.Tensor.sum is not torch._C.TensorBase.sum
        or torch.Tensor.to is not torch._C.TensorBase.to
        or torch.Tensor.new_empty is not torch._C.TensorBase.new_empty
    ):
        return None
    if not _ordinary_tensor(left) or not _ordinary_tensor(right):
        return None
    if left.device.type != "cuda" or right.device.type != "cuda":
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
    plan = plan_for_pair(left, right, dtype, layout)
    if plan is None or not _compatible_aten_sum():
        return None
    # Import CuTe only after the metadata and arithmetic compatibility proof.
    from ..._compiler.cute.paired_sum_runtime import paired_sum_cast_kernel

    left_out = left.new_empty((plan.columns,), dtype=dtype)
    right_out = right.new_empty((plan.columns,), dtype=dtype)
    if plan.columns:
        _launcher(
            paired_sum_cast_kernel,
            plan.grid,
            left,
            right,
            left_out,
            right_out,
            plan.rows,
            plan.columns,
            plan.row_lanes,
            plan.column_threads,
            plan.vector,
            block=plan.block,
        )
    return left_out, right_out
