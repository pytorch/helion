"""Isolate the emitted split merge's FP32 rounding from its BF16 matmul producer."""

from __future__ import annotations

import importlib

import cutlass
from cutlass import cute
from cutlass.cute.runtime import from_dlpack
import torch

from helion._compiler.cute._gathered_softmax_runtime import split_weighted_merge

cuda = importlib.import_module("cuda.bindings.driver")


@cute.kernel
def _merge_one(
    scores: cute.Tensor,
    partials: cute.Tensor,
    out: cute.Tensor,
    live: cutlass.Int32,
    fast_math: cutlass.Constexpr,
) -> None:
    split_weighted_merge(
        scores,
        partials,
        out,
        cutlass.Int32(0),
        cutlass.Int32(0),
        live,
        fast_math=fast_math,
    )


@cute.jit
def _launch_merge_one(
    scores: cute.Tensor,
    partials: cute.Tensor,
    out: cute.Tensor,
    live: cutlass.Int32,
    fast_math: cutlass.Constexpr,
    stream: cuda.CUstream,
) -> None:
    _merge_one(scores, partials, out, live, fast_math).launch(
        grid=(1, 1, 1), block=(128, 1, 1), smem=272, stream=stream
    )


def merge_one(
    scores: torch.Tensor,
    partials: torch.Tensor,
    output: torch.Tensor,
    live: int,
    fast_math: bool,
) -> None:
    _launch_merge_one(
        from_dlpack(scores),
        from_dlpack(partials),
        from_dlpack(output),
        cutlass.Int32(live),
        fast_math,
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
