"""
FP8 RowWise Scaled Matrix Multiplication with Helion
====================================================
This example implements an FP8 (e4m3) RowWise ``scaled_mm`` kernel in Helion:

    out[m, n] = scale_a[m] * scale_b[n] * sum_k a[m, k] * b[k, n]

The rowwise scale is *linear/separable* across the K dimension, so it can be
folded into each split-K partial before an atomic accumulation. This lets a
single split-K kernel (no separate reduction kernel) cover the skinny-M regime
(small M, large N/K) that is HBM-bandwidth bound, while filling the machine via
split-K. The accumulator is FP32; results are atomically added into a pre-zeroed
BF16 output.

A reference using ``torch._scaled_mm`` validates correctness.
"""

# %%
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import helion
from helion._testing import DEVICE
from helion.autotuner import PowerOfTwoFragment
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable


# %%
@helion.kernel(static_shapes=True)
def scaled_mm(
    x: torch.Tensor,
    y: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    """
    FP8 RowWise scaled matrix multiplication using split-K parallelism.

    Args:
        x (torch.Tensor): Left input matrix of shape [m, k] in FP8 (e4m3).
        y (torch.Tensor): Right input matrix of shape [k, n] in FP8 (e4m3).
        scale_a (torch.Tensor): Per-row scale of shape [m, 1].
        scale_b (torch.Tensor): Per-column scale of shape [1, n].

    Returns:
        torch.Tensor: Output matrix of shape [m, n] in BF16.
    """
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f"size mismatch {k} != {k2}"
    split_k = hl.register_tunable("split_k", PowerOfTwoFragment(1, 256))
    # When split_k == 1 every output tile is written exactly once, so we can use
    # a plain store into an uninitialized buffer -- no atomics and, crucially,
    # no output memset (which otherwise costs ~0.7-1.2us and is the whole gap to
    # vLLM at moderate M). When split_k > 1 the reduction is split across CTAs
    # to fill the machine on very skinny M, and partials are accumulated with
    # atomic_add into a pre-zeroed output.
    if split_k == 1:
        out = torch.empty([m, n], dtype=torch.bfloat16, device=x.device)
    else:
        out = torch.zeros([m, n], dtype=torch.bfloat16, device=x.device)
    k_block = helion.next_power_of_2(helion.cdiv(k, split_k))
    for tile_m, tile_n, outer_k in hl.tile([m, n, k], block_size=[None, None, k_block]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for inner_k in hl.tile(outer_k.begin, outer_k.end):
            acc = hl.dot(x[tile_m, inner_k], y[inner_k, tile_n], acc=acc)
        # Fold the rowwise scale into this K-split's partial (scale is linear
        # across K, so summing the scaled partials yields the scaled total).
        acc = acc * scale_a[tile_m, :] * scale_b[:, tile_n]
        if split_k == 1:
            out[tile_m, tile_n] = acc.to(torch.bfloat16)
        else:
            hl.atomic_add(out, [tile_m, tile_n], acc.to(torch.bfloat16))
    return out


# %%
@helion.kernel(static_shapes=True)
def scaled_mm_into(
    out: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    """
    Split-K FP8 RowWise scaled_mm that accumulates into a caller-provided,
    pre-zeroed ``out``. Unlike :func:`scaled_mm`, this performs no internal
    allocation or memset, so the ~0.7-1.2us output-zeroing tax can be hoisted
    out of the timed region and overlapped on a separate CUDA stream with the
    previous call's compute (double-buffered "ping-pong"). With that overlap
    the kernel reaches its memset-free compute floor, matching vLLM.

    Args:
        out (torch.Tensor): Pre-zeroed output of shape [m, n] in BF16; mutated.
        x (torch.Tensor): Left input matrix of shape [m, k] in FP8 (e4m3).
        y (torch.Tensor): Right input matrix of shape [k, n] in FP8 (e4m3).
        scale_a (torch.Tensor): Per-row scale of shape [m, 1].
        scale_b (torch.Tensor): Per-column scale of shape [1, n].
    """
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f"size mismatch {k} != {k2}"
    split_k = hl.register_tunable("split_k", PowerOfTwoFragment(2, 256))
    k_block = helion.next_power_of_2(helion.cdiv(k, split_k))
    for tile_m, tile_n, outer_k in hl.tile([m, n, k], block_size=[None, None, k_block]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for inner_k in hl.tile(outer_k.begin, outer_k.end):
            acc = hl.dot(x[tile_m, inner_k], y[inner_k, tile_n], acc=acc)
        acc = acc * scale_a[tile_m, :] * scale_b[:, tile_n]
        hl.atomic_add(out, [tile_m, tile_n], acc.to(torch.bfloat16))
    return out


# %%
@helion.kernel(static_shapes=True)
def scaled_mm_compute(
    x: torch.Tensor,
    y: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    """
    Plain (no split-K, no atomics) FP8 RowWise scaled_mm for the *compute-bound*
    regime (large M, or large N/K). Each output tile is computed by one program
    and stored once, so Helion's autotuner is free to explore the full Blackwell
    GEMM recipe (TMA loads, warp-specialized persistent mainloop, epilogue
    subtiling, deep pipelining) without the split-K/atomic structure getting in
    the way. Use full autotuning (HELION_AUTOTUNE_EFFORT=full) for these shapes.

    Args:
        x (torch.Tensor): Left input matrix of shape [m, k] in FP8 (e4m3).
        y (torch.Tensor): Right input matrix of shape [k, n] in FP8 (e4m3).
        scale_a (torch.Tensor): Per-row scale of shape [m, 1].
        scale_b (torch.Tensor): Per-column scale of shape [1, n].

    Returns:
        torch.Tensor: Output matrix of shape [m, n] in BF16.
    """
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f"size mismatch {k} != {k2}"
    out = torch.empty([m, n], dtype=torch.bfloat16, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
        acc = acc * scale_a[tile_m, :] * scale_b[:, tile_n]
        out[tile_m, tile_n] = acc.to(torch.bfloat16)
    return out


# %%
def reference_scaled_mm(
    x: torch.Tensor,
    y: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    """Reference using ``torch._scaled_mm`` (column-major second operand)."""
    if y.stride(0) == 1 and y.stride(1) > 1:
        y_col_major = y
    else:
        y_col_major = y.T.contiguous().T
    return torch._scaled_mm(
        x, y_col_major, scale_a, scale_b, use_fast_accum=False, out_dtype=torch.bfloat16
    )


# %%
def scaled_mm_tritonbench(
    tb_op: object,
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> Callable[[], torch.Tensor]:
    """Wrapper for TritonBench compatibility.

    tritonbench passes ``a`` [m, k], ``b`` [k, n] (col-major), and rowwise
    scales ``scale_a`` [m, 1], ``scale_b`` [1, n].
    """
    return lambda: scaled_mm(a, b, scale_a, scale_b)


# %%
def check(m: int, k: int, n: int) -> None:
    """Validate the scaled_mm kernel against the torch._scaled_mm reference."""
    x = torch.randn([m, k], device=DEVICE, dtype=torch.float32)
    y = torch.randn([k, n], device=DEVICE, dtype=torch.float32)
    x_fp8 = (x * 0.4).to(torch.float8_e4m3fn)
    y_fp8 = (y * 0.4).to(torch.float8_e4m3fn).T.contiguous().T  # col-major
    scale_a = (torch.rand([m, 1], device=DEVICE) + 0.5).to(torch.float32)
    scale_b = (torch.rand([1, n], device=DEVICE) + 0.5).to(torch.float32)

    from helion._testing import run_example

    run_example(
        lambda a, b: scaled_mm(a, b, scale_a, scale_b),
        lambda a, b: reference_scaled_mm(a, b, scale_a, scale_b),
        (x_fp8, y_fp8),
        atol=0.1,
        rtol=0.1,
    )


# %%
shapes = [  # m, k, n  (skinny-M FP8 decode shapes)
    (1, 4096, 4096),
    (16, 4096, 4096),
    (1, 4096, 14336),
    (16, 4096, 14336),
]


# %%
# Qwen3-1.7B FP8 (W8A8: per-channel weight + per-token dynamic act) linear-layer
# GEMMs as seen by vllm_ops.cutlass_scaled_mm during vLLM serving benchmarks
# (input 512 / output 600, batch-size sweep 1..64). (K, N) are fixed by the model;
# lm_head is excluded (kept bf16). M is the token count per forward:
#   decode -> running batch size {1,2,4,8,16,32,64}; prefill -> ~512-token chunk.
_QWEN3_1_7B_LAYER_KN = [  # (k, n) per decoder layer (x28 layers)
    (2048, 4096),    # qkv_proj      (q 2048 + k 1024 + v 1024, GQA)
    (2048, 2048),    # o_proj
    (2048, 12288),   # gate_up_proj  (2 x intermediate 6144, fused)
    (6144, 2048),    # down_proj
]
_QWEN3_1_7B_M = [1, 2, 4, 8, 16, 32, 64, 512]  # decode batch sizes + 512-token prefill

vllm_shapes = [  # m, k, n
    (m, k, n) for m in _QWEN3_1_7B_M for (k, n) in _QWEN3_1_7B_LAYER_KN
]


# %%
def main() -> None:
    """Run correctness checks across the skinny-M shapes."""
    for m, k, n in shapes:
        print(f"Testing scaled_mm shape=(m={m}, k={k}, n={n})")
        check(m, k, n)


# %%
if __name__ == "__main__":
    main()
