"""Packed MXFP8 dense GEMM for Blackwell (sm100/sm103) -- Helion CuTe backend.

The CuTe counterpart of ``pretuned_kernels/mxfp8_gemm`` (the Triton variant):
the same kernel body, inputs, packed MN-major E8M0 scale words and baselines,
compiled with ``backend="cute"``.  The CuTe backend proves the registered task
grid and the complete packed-scale K reduction, and lowers the contraction to
tcgen05 block-scaled MMAs with TMA operand and scale-factor loads
(``cute_scaled_mma``); the checked-in configs pick its tile, pipeline depth and
cluster shape per shape.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import torch

import helion
from helion._testing import import_path
from helion.autotuner import PowerOfTwoFragment
import helion.language as hl

_triton_variant = import_path(
    Path(__file__).resolve().parents[1] / "mxfp8_gemm" / "mxfp8_gemm.py"
)
SHAPES = _triton_variant.SHAPES
baselines = _triton_variant.baselines
make_inputs = _triton_variant.make_inputs
pack_mxfp8_scales = _triton_variant.pack_mxfp8_scales
reference_mxfp8_gemm = _triton_variant.reference_mxfp8_gemm


@helion.aot_kernel(
    backend="cute",
    static_shapes=True,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def mxfp8_gemm_cute(
    a: torch.Tensor,  # [M, K] float8_e4m3fn
    b: torch.Tensor,  # [N, K] float8_e4m3fn
    scale_a: torch.Tensor,  # [M, ceil(K / 128)] int32, MN-major packed E8M0
    scale_b: torch.Tensor,  # [N, ceil(K / 128)] int32, MN-major packed E8M0
) -> torch.Tensor:
    """Return BF16 ``dequant(a) @ dequant(b).T`` ``[M, N]``."""
    m, k = a.shape
    n = b.size(0)
    assert b.size(1) == k and k % 32 == 0
    assert a.dtype == b.dtype == torch.float8_e4m3fn
    assert scale_a.dtype == scale_b.dtype == torch.int32
    out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    block_m = hl.register_block_size(16, 128)
    block_n = hl.register_block_size(16, 256)
    block_groups = hl.register_block_size(2, 32)
    split_k = hl.register_tunable("split_k", PowerOfTwoFragment(1, 16, 1))
    groups_per_split = (k // 32 + split_k - 1) // split_k
    partial = torch.empty((split_k, m, n), dtype=torch.float32, device=a.device)
    flat_out = out.view(m * n)
    flat_partial = partial.view(split_k * m * n)
    for task_m, task_n, part in hl.grid(
        [(m + block_m - 1) // block_m, (n + block_n - 1) // block_n, split_k]
    ):
        rows = task_m * block_m + hl.arange(block_m)
        cols = task_n * block_n + hl.arange(block_n)
        acc = hl.zeros([block_m, block_n], dtype=torch.float32)
        group_end = min((part + 1) * groups_per_split, k // 32)
        for groups in hl.tile(
            part * groups_per_split, group_end, block_size=block_groups
        ):
            columns = groups.begin * 32 + hl.arange(block_groups * 32)
            lhs = hl.load(
                a,
                [rows, columns],
                extra_mask=(rows[:, None] < m) & (columns[None, :] < group_end * 32),
            )
            rhs = hl.load(
                b,
                [cols, columns],
                extra_mask=(cols[:, None] < n) & (columns[None, :] < group_end * 32),
            )
            lhs_words = hl.load(
                scale_a,
                [rows, groups.index // 4],
                extra_mask=(rows[:, None] < m) & (groups.index[None, :] < group_end),
            )
            rhs_words = hl.load(
                scale_b,
                [cols, groups.index // 4],
                extra_mask=(cols[:, None] < n) & (groups.index[None, :] < group_end),
            )
            shift = ((groups.index % 4) * 8)[None, :]
            lhs_scale = ((lhs_words >> shift) & 255).to(torch.uint8)
            rhs_scale = ((rhs_words >> shift) & 255).to(torch.uint8)
            acc = hl.dot_scaled(
                lhs, lhs_scale, "e4m3", rhs.T, rhs_scale, "e4m3", acc=acc
            )
        if split_k == 1:
            hl.store(
                flat_out,
                [rows[:, None] * n + cols[None, :]],
                acc.to(out.dtype),
                extra_mask=(rows[:, None] < m) & (cols[None, :] < n),
            )
        else:
            hl.store(
                flat_partial,
                [(part * m + rows[:, None]) * n + cols[None, :]],
                acc,
                extra_mask=(rows[:, None] < m) & (cols[None, :] < n),
            )
    if split_k > 1:
        return partial.sum(dim=0).to(torch.bfloat16)
    return out


def use_cudagraph() -> bool:
    """Whether main() benchmarks under CUDA graphs (read by pretuned_kernels/run.py)."""
    return True


def correctness_check(shapes: list[tuple[int, int, int]] | None = None) -> None:
    """Check the CuTe kernel against the FP32 dequantization reference."""
    _triton_variant.correctness_check(mxfp8_gemm_cute, shapes)


def main(verbose: bool = True) -> dict:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return _triton_variant.main(verbose=verbose, kernel=mxfp8_gemm_cute)


if __name__ == "__main__":
    main()
