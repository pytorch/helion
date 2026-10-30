"""Matmul performance sweep: Helion CuTe vs Helion Triton vs torch (cuBLAS).

About a hundred GEMM-family cells (plain, bias / relu / gelu epilogues, bmm,
broadcast bmm, split-K, fp8 tensor-wise and row-wise, bf16 x int4, bf16 x int16,
matmul + layernorm, jagged grouped) over square, long, wide, skinny, K-heavy,
odd-sized and transposed-layout shapes in fp16 / bf16 / fp32(tf32) / fp8.

Every implementation is timed the same way, from the same inputs to its
output, in three regimes: ``steady_ms`` is the sustained regime the
autotuner's final selection and a deployed kernel sequence run in (three
seconds of back-to-back flushed calls drive the power-capped GPU to its
sustained clock, then the median of five 500 ms + 500 ms flushed steady
windows); ``bench_ms`` is the autotuner's cold-L2 burst regime
(``interleaved_bench``: one CUDA-graph replay of [L2 flush; call], flush
subtracted, median of five 100-sample runs at boost clocks); ``event_ms`` is
a warm ``do_bench`` median. Ratios prefer ``steady_ms`` when both records
have it (``summarize --timer`` picks one explicitly). Compiled torch
baselines are compiled before timing. Autotunes are cold: fresh random seed,
private cache directories, ``HELION_FORCE_AUTOTUNE=1``, and the autotuner
ranks in the cold-L2 ``call`` regime.

  python benchmarks/cute/matmul_sweep.py list [--cells REGEX]
  python benchmarks/cute/matmul_sweep.py run --out artifacts/<run>/quick \\
      --gpus 0,1,2,3 --providers cute,triton,torch --effort quick
  python benchmarks/cute/matmul_sweep.py worker --cell mm-bf16-nn-4096x4096x4096 \\
      --provider cute --effort full --json out.json
  python benchmarks/cute/matmul_sweep.py summarize --out artifacts/<run>/quick
  python benchmarks/cute/matmul_sweep.py abab --cell <id> --pin-from <cute record> \\
      --json out.json       # interleaved cute-vs-torch rounds in one process
  python benchmarks/cute/matmul_sweep.py run --out <dir> --gpus 4 --providers cute \\
      --pin-dir artifacts/<run>/full   # re-time the configs recorded there

Cell ids are ``<family>-<dtype>-<layout>-<shape>``; shapes read ``MxNxK``
(``BxMxNxK`` for batched families). ``ratio`` in the summary is
``torch_ms / cute_ms`` (>1 means Helion CuTe is faster).
"""

# ruff: noqa: ANN401
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import os
from pathlib import Path
import re
import secrets
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from typing import Any
from typing import Callable
from typing import cast

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))

PROVIDERS = ("cute", "triton", "torch")


# --------------------------------------------------------------------------
# Cells
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Cell:
    family: str
    dtype: str
    layout: str
    shape: tuple[int, ...]
    note: str = ""
    # Tolerances for the correctness check against the torch reference.
    rtol: float = 1e-2
    atol: float = 1e-1
    # Fraction of elements allowed outside the tolerance (0 = assert_close).
    mismatch_budget: float = 0.0

    @property
    def id(self) -> str:
        return (
            f"{self.family}-{self.dtype}-{self.layout}-{'x'.join(map(str, self.shape))}"
        )

    @property
    def flops(self) -> float:
        if self.family in ("bmm", "bcast"):
            b, m, n, k = self.shape
            return 2.0 * b * m * n * k
        m, n, k = self.shape[:3]
        return 2.0 * m * n * k


def _mm_cells() -> list[Cell]:
    cells: list[Cell] = []

    def mm(dtype: str, layout: str, m: int, n: int, k: int, note: str) -> None:
        if dtype == "fp32":
            # Both sides run tf32 (10-bit mantissa products): the backends
            # round operands and order the fp32 sums differently.
            cells.append(
                Cell(
                    "mm", dtype, layout, (m, n, k), note, atol=0.5, mismatch_budget=1e-3
                )
            )
            return
        cells.append(Cell("mm", dtype, layout, (m, n, k), note))

    # Square, powers of two.
    for s in (256, 512, 1024, 2048, 4096, 8192):
        mm("bf16", "nn", s, s, s, "square")
    # Odd / ragged sizes (partial tiles, unaligned K, tiny).
    mm("bf16", "nn", 1000, 1000, 1000, "odd square")
    mm("bf16", "nn", 3000, 3000, 3000, "odd square")
    mm("bf16", "nn", 4097, 4097, 4097, "pow2+1 square")
    mm("bf16", "nn", 777, 999, 555, "odd all dims, K not a multiple of 8")
    mm("bf16", "nn", 1024, 1024, 1000, "K not a multiple of 16")
    mm("bf16", "nn", 2048, 2048, 2050, "K = pow2 + 2")
    mm("bf16", "nn", 65, 128, 32, "partial M tile, tiny K")
    mm("bf16", "nn", 33, 33, 33, "tiny odd")
    # Long (tall-skinny output in M).
    mm("bf16", "nn", 65536, 512, 512, "tall")
    mm("bf16", "nn", 32768, 1024, 1024, "tall")
    mm("bf16", "nn", 16384, 256, 4096, "tall, narrow N")
    mm("bf16", "nn", 131072, 128, 128, "very tall, tiny N/K")
    # Wide.
    mm("bf16", "nn", 512, 65536, 512, "wide")
    mm("bf16", "nn", 256, 32768, 4096, "wide, small M")
    mm("bf16", "nn", 128, 131072, 128, "very wide, tiny M/K")
    # Skinny (decode-like, memory bound).
    mm("bf16", "nn", 1, 4096, 4096, "gemv")
    mm("bf16", "nn", 8, 4096, 4096, "skinny M=8")
    mm("bf16", "nn", 16, 14336, 4096, "skinny M=16 MLP up")
    mm("bf16", "nn", 32, 4096, 14336, "skinny M=32 MLP down")
    mm("bf16", "nn", 64, 8192, 8192, "skinny M=64")
    # K-heavy (split-K territory).
    mm("bf16", "nn", 512, 512, 65536, "K-heavy")
    mm("bf16", "nn", 1024, 1024, 32768, "K-heavy")
    mm("bf16", "nn", 2048, 2048, 16384, "K-heavy")
    mm("bf16", "nn", 256, 256, 131072, "K-heavy, tiny MN")
    # LLM shapes (Llama-class projections, lm head).
    mm("bf16", "nn", 2048, 14336, 4096, "MLP up")
    mm("bf16", "nn", 2048, 4096, 14336, "MLP down")
    mm("bf16", "nn", 8192, 1024, 4096, "GQA kv proj")
    mm("bf16", "nn", 16384, 4096, 4096, "long prefill qkv")
    mm("bf16", "nn", 4096, 128256, 4096, "lm head")
    mm("bf16", "nn", 1024, 8192, 28672, "wide-K MLP down")
    # Transposed operand layouts.
    mm("bf16", "nt", 4096, 4096, 4096, "B K-contiguous (nn.Linear weight)")
    mm("bf16", "tn", 4096, 4096, 4096, "A M-contiguous")
    mm("bf16", "tt", 4096, 4096, 4096, "both transposed")
    mm("bf16", "nt", 2048, 14336, 4096, "MLP up, Linear layout")
    mm("bf16", "nt", 1000, 1000, 1000, "odd, Linear layout")
    mm("bf16", "nt", 16, 4096, 4096, "skinny, Linear layout")
    # fp16.
    mm("fp16", "nn", 1024, 1024, 1024, "square")
    mm("fp16", "nn", 4096, 4096, 4096, "square")
    mm("fp16", "nn", 8192, 8192, 8192, "square")
    mm("fp16", "nt", 2048, 14336, 4096, "MLP up, Linear layout")
    # fp32 (tf32 on both sides).
    mm("fp32", "nn", 1024, 1024, 1024, "square tf32")
    mm("fp32", "nn", 4096, 4096, 4096, "square tf32")
    mm("fp32", "nn", 8192, 8192, 8192, "square tf32")
    mm("fp32", "nt", 2048, 2048, 2048, "Linear layout tf32")
    return cells


def _epilogue_cells() -> list[Cell]:
    cells: list[Cell] = []
    # matmul + 1-D bias (nn.Linear): torch baseline F.linear (cuBLASLt epilogue).
    for m, n, k, note in (
        (4096, 4096, 4096, "square"),
        (2048, 14336, 4096, "MLP up"),
        (16, 4096, 4096, "skinny"),
        (1000, 1000, 1000, "odd"),
    ):
        cells.append(Cell("linear", "bf16", "nt", (m, n, k), note))
    # matmul + bias + relu: torch baseline _addmm_activation (cuBLASLt epilogue).
    for m, n, k in ((4096, 4096, 4096), (1024, 1024, 1024)):
        cells.append(Cell("relu", "bf16", "nt", (m, n, k), "bias+relu"))
    # epilogue_subtiling example: scale, residual, bias, gelu, cast.
    for m, n, k in ((1024, 1024, 1024), (4096, 4096, 4096), (2048, 2048, 4096)):
        cells.append(Cell("gelu", "fp16", "nn", (m, n, k), "scale+residual+bias+gelu"))
    cells.append(Cell("geluaux", "fp16", "nn", (4096, 4096, 4096), "gelu + aux out"))
    # matmul + layernorm.
    for m, n, k in ((512, 1024, 1024), (2048, 2048, 2048), (4096, 1024, 4096)):
        cells.extend(
            (
                Cell("mmln", "fp16", "nn", (m, n, k), "matmul+layernorm"),
                Cell(
                    "mmln2",
                    "fp16",
                    "nn",
                    (m, n, k),
                    "matmul then layernorm (2 launches)",
                ),
            )
        )
    return cells


def _batched_cells() -> list[Cell]:
    cells: list[Cell] = []
    for b, m, n, k, layout, note in (
        (4, 64, 128, 128, "nn", "tiny"),
        (8, 256, 256, 512, "nn", "small"),
        (16, 512, 768, 1024, "nn", "medium"),
        (32, 1024, 1024, 1024, "nn", "large"),
        (128, 64, 64, 64, "nn", "many tiny"),
        (8, 4096, 4096, 128, "nn", "QK^T-like"),
        (64, 128, 128, 256, "nt", "B K-contiguous"),
        (16, 1000, 1000, 1000, "nn", "odd"),
    ):
        cells.append(Cell("bmm", "bf16", layout, (b, m, n, k), note))
    for b, m, n, k, note in (
        (8, 256, 256, 512, "small"),
        (16, 512, 768, 1024, "medium"),
        (32, 128, 4096, 4096, "proj"),
    ):
        cells.append(Cell("bcast", "bf16", "nn", (b, m, n, k), note))
    return cells


def _splitk_cells() -> list[Cell]:
    # The example accumulates its split-K partial sums into the bf16 output
    # with atomics, so every split adds a bf16 rounding: a few tenths of a
    # percent of the outputs land 2-4 ulps (8.0 at |sum| ~ 500) from torch,
    # more so with the deeper splits the CuTe search picks (0.35-0.45%).
    return [
        Cell(
            "splitk",
            "bf16",
            "nn",
            (256, 256, 65536),
            "K-heavy",
            atol=1.0,
            mismatch_budget=1e-2,
        ),
        Cell(
            "splitk",
            "bf16",
            "nn",
            (1024, 1024, 32768),
            "K-heavy",
            atol=1.0,
            mismatch_budget=1e-2,
        ),
        Cell(
            "splitk",
            "bf16",
            "nn",
            (512, 512, 131072),
            "K-heavy",
            atol=1.0,
            mismatch_budget=1e-2,
        ),
    ]


def _fp8_cells() -> list[Cell]:
    cells: list[Cell] = []
    for m, n, k, layout, note in (
        (1024, 1024, 1024, "nt", "square"),
        (4096, 4096, 4096, "nt", "square"),
        (8192, 8192, 8192, "nt", "square"),
        (2048, 14336, 4096, "nt", "MLP up"),
        (16, 4096, 4096, "nt", "skinny"),
        (4096, 4096, 4096, "nn", "B N-contiguous (row-major)"),
    ):
        cells.append(Cell("fp8", "e4m3", layout, (m, n, k), note, rtol=5e-2, atol=0.5))
    for m, n, k in ((1024, 1024, 1024), (4096, 4096, 4096), (2048, 14336, 4096)):
        cells.append(
            Cell(
                "fp8rw", "e4m3", "nt", (m, n, k), "rowwise scales", rtol=5e-2, atol=0.5
            )
        )
    return cells


def _quantized_cells() -> list[Cell]:
    cells: list[Cell] = []
    for m, n, k, note in (
        (16, 4096, 4096, "decode"),
        (1024, 4096, 4096, "prefill"),
        (4096, 11008, 4096, "prefill MLP"),
    ):
        cells.append(Cell("int4", "bf16w4", "nn", (m, n, k), note, rtol=2e-1, atol=1.0))
    for m, n, k, layout, note in (
        (512, 512, 1024, "nn", "small"),
        (2048, 4096, 4096, "nn", "medium"),
        (2048, 4096, 4096, "t", "int16 activations"),
    ):
        cells.append(
            Cell(
                "i16",
                "bf16i16",
                layout,
                (m, n, k),
                note,
                # int16 operands reach |sum| ~ 1e6-1e8: fp32 accumulation
                # order flips bf16 rounding on a handful of outputs.
                rtol=2e-2,
                atol=1e-2,
                mismatch_budget=1e-3,
            )
        )
    return cells


def _grouped_cells() -> list[Cell]:
    # shape = (total_M, N, K); the group split is derived from the note.
    return [
        Cell("grouped", "bf16", "g4", (640, 128, 256), "4 groups 64..256 rows"),
        Cell("grouped", "bf16", "g8", (8192, 4096, 4096), "8 groups, mixed rows"),
        Cell("grouped", "bf16", "g16", (16384, 8192, 2048), "16 ragged groups"),
        Cell("groupedp", "bf16", "g8", (8192, 4096, 4096), "persistent, 8 groups"),
        # The shared-weight formulation (one tile space over the packed rows,
        # persistence left to the autotuner) of the same problems.
        Cell("groupedv2", "bf16", "g8", (8192, 4096, 4096), "shared-weight kernel"),
        Cell("groupedv2", "bf16", "g16", (16384, 8192, 2048), "shared-weight kernel"),
    ]


def all_cells() -> list[Cell]:
    cells = (
        _mm_cells()
        + _epilogue_cells()
        + _batched_cells()
        + _splitk_cells()
        + _fp8_cells()
        + _quantized_cells()
        + _grouped_cells()
    )
    ids = [c.id for c in cells]
    assert len(ids) == len(set(ids)), "duplicate cell ids"
    return cells


def select_cells(pattern: str | None) -> list[Cell]:
    cells = all_cells()
    if pattern:
        regex = re.compile(pattern)
        cells = [c for c in cells if regex.search(c.id)]
    return cells


def cell_by_id(cell_id: str) -> Cell:
    for cell in all_cells():
        if cell.id == cell_id:
            return cell
    raise KeyError(cell_id)


# --------------------------------------------------------------------------
# Building the implementations of one cell
# --------------------------------------------------------------------------


@dataclasses.dataclass
class Built:
    helion_fn: Callable[[], Any] | None
    torch_fn: Callable[[], Any]
    reference_fn: Callable[[], Any]
    kernels: list[Any]
    torch_impl: str
    inputs: dict[str, Any]


_DTYPES = {
    "fp16": "float16",
    "bf16": "bfloat16",
    "fp32": "float32",
}


def _torch_dtype(name: str) -> torch.dtype:
    return getattr(torch, _DTYPES[name])


def _operand(
    shape: tuple[int, int], dtype: torch.dtype, transposed: bool
) -> torch.Tensor:
    """Row-major tensor of ``shape``, or a transposed view with the same logical shape."""
    if transposed:
        return torch.randn(shape[::-1], device="cuda", dtype=dtype).T
    return torch.randn(shape, device="cuda", dtype=dtype)


def _group_rows(total: int, groups: int, seed: int) -> list[int]:
    if groups == 4:
        return [64 * (i + 1) for i in range(groups)]
    gen = torch.Generator().manual_seed(seed)
    weights = torch.rand(groups, generator=gen) + 0.25
    rows = (weights / weights.sum() * total).floor().to(torch.int64)
    rows = (rows // 8 * 8).tolist()
    rows[-1] = total - sum(rows[:-1])
    return rows


def build(cell: Cell, provider: str) -> Built:
    import torch.nn.functional as F

    torch.manual_seed(0)
    helion_fn: Callable[[], Any] | None = None
    kernels: list[Any] = []
    inputs: dict[str, Any] = {}
    fam = cell.family
    want_helion = provider in ("cute", "triton")

    if fam in ("mm", "linear", "relu"):
        m, n, k = cell.shape
        dtype = _torch_dtype(cell.dtype)
        x = _operand((m, k), dtype, cell.layout[0] == "t")
        y = _operand((k, n), dtype, cell.layout[1] == "t")
        inputs = {"x": x, "y": y}
        if dtype == torch.float32:
            torch.backends.cuda.matmul.allow_tf32 = True
        if fam == "mm":
            torch_fn = lambda: torch.matmul(x, y)  # noqa: E731
            torch_impl = "torch.matmul (cuBLAS)"
            reference_fn = torch_fn
        else:
            bias = torch.randn(n, device="cuda", dtype=dtype)
            inputs["bias"] = bias
            if fam == "linear":
                w = y.T  # [n, k]
                torch_fn = lambda: F.linear(x, w, bias)  # noqa: E731
                torch_impl = "F.linear (cuBLASLt bias epilogue)"
                reference_fn = lambda: torch.matmul(x, y) + bias  # noqa: E731
            else:
                torch_fn = lambda: torch._addmm_activation(bias, x, y)  # noqa: E731
                torch_impl = "torch._addmm_activation (cuBLASLt relu epilogue)"
                reference_fn = lambda: torch.relu(torch.matmul(x, y) + bias)  # noqa: E731
        if want_helion:
            from examples.matmul import matmul

            kernels = [matmul]
            if fam == "mm":
                helion_fn = lambda: matmul(x, y)  # noqa: E731
            elif fam == "linear":
                helion_fn = lambda: matmul(x, y, lambda acc, tile: acc + bias[tile[1]])  # noqa: E731
            else:
                helion_fn = lambda: matmul(  # noqa: E731
                    x, y, lambda acc, tile: torch.relu(acc + bias[tile[1]])
                )
    elif fam in ("gelu", "geluaux"):
        m, n, k = cell.shape
        dtype = _torch_dtype(cell.dtype)
        x = torch.randn(m, k, device="cuda", dtype=dtype)
        w = torch.randn(k, n, device="cuda", dtype=dtype)
        bias = torch.randn(n, device="cuda", dtype=dtype)
        residual = torch.randn(m, n, device="cuda", dtype=dtype)
        inputs = {"x": x, "w": w, "bias": bias, "residual": residual}
        if fam == "gelu":

            def ref_gelu() -> torch.Tensor:
                acc = (x @ w).float()
                val = acc * 1.25 + residual.float() * 0.5 + bias.float()
                return F.gelu(val).to(dtype)

            reference_fn = ref_gelu
            torch_fn = torch.compile(ref_gelu)
            torch_impl = "torch.compile(bf16 mm + fused fp32 epilogue)"
        else:

            def ref_aux() -> tuple[torch.Tensor, torch.Tensor]:
                acc = (x @ w).float()
                pre = acc * 1.25 + bias.float()
                return F.gelu(pre).to(dtype), pre.to(dtype)

            reference_fn = ref_aux
            torch_fn = torch.compile(ref_aux)
            torch_impl = "torch.compile(bf16 mm + fused fp32 epilogue, 2 outputs)"
        if want_helion:
            from examples.epilogue_subtiling import matmul_bias_gelu_aux
            from examples.epilogue_subtiling import matmul_bias_residual_gelu_cast

            if fam == "gelu":
                kernels = [matmul_bias_residual_gelu_cast]
                helion_fn = lambda: matmul_bias_residual_gelu_cast(x, w, bias, residual)  # noqa: E731
            else:
                kernels = [matmul_bias_gelu_aux]
                helion_fn = lambda: matmul_bias_gelu_aux(x, w, bias)  # noqa: E731
    elif fam in ("mmln", "mmln2"):
        m, n, k = cell.shape
        dtype = _torch_dtype(cell.dtype)
        x = torch.randn(m, k, device="cuda", dtype=dtype)
        y = torch.randn(k, n, device="cuda", dtype=dtype)
        weight = torch.randn(n, device="cuda", dtype=dtype)
        bias = torch.randn(n, device="cuda", dtype=dtype)
        inputs = {"x": x, "y": y, "weight": weight, "bias": bias}
        from examples.matmul_layernorm import matmul_layernorm_pytorch

        reference_fn = lambda: matmul_layernorm_pytorch(x, y, weight, bias)  # noqa: E731
        torch_fn = torch.compile(reference_fn)
        torch_impl = "torch.compile(matmul + layer_norm)"
        if want_helion:
            if fam == "mmln":
                from examples.matmul_layernorm import matmul_layernorm

                kernels = [matmul_layernorm]
                helion_fn = lambda: matmul_layernorm(x, y, weight, bias)  # noqa: E731
            else:
                from examples.matmul import matmul
                from examples.matmul_layernorm import layernorm_rows
                from examples.matmul_layernorm import matmul_layernorm_unfused

                kernels = [matmul, layernorm_rows]
                helion_fn = lambda: matmul_layernorm_unfused(x, y, weight, bias)  # noqa: E731
    elif fam == "bmm":
        b, m, n, k = cell.shape
        dtype = _torch_dtype(cell.dtype)
        A = torch.randn(b, m, k, device="cuda", dtype=dtype)
        if cell.layout == "nt":
            B = torch.randn(b, n, k, device="cuda", dtype=dtype).transpose(1, 2)
        else:
            B = torch.randn(b, k, n, device="cuda", dtype=dtype)
        inputs = {"A": A, "B": B}
        torch_fn = lambda: torch.bmm(A, B)  # noqa: E731
        torch_impl = "torch.bmm (cuBLAS)"
        reference_fn = torch_fn
        if want_helion:
            from examples.bmm import bmm

            kernels = [bmm]
            helion_fn = lambda: bmm(A, B)  # noqa: E731
    elif fam == "bcast":
        b, m, n, k = cell.shape
        dtype = _torch_dtype(cell.dtype)
        x = torch.randn(b, m, k, device="cuda", dtype=dtype)
        w = torch.randn(k, n, device="cuda", dtype=dtype)
        inputs = {"x": x, "w": w}
        torch_fn = lambda: torch.matmul(x, w)  # noqa: E731
        torch_impl = "torch.matmul (cuBLAS, folded batch)"
        reference_fn = torch_fn
        if want_helion:
            from examples.broadcast_matmul import broadcast_matmul

            kernels = [broadcast_matmul]
            helion_fn = lambda: broadcast_matmul(x, w)  # noqa: E731
    elif fam == "splitk":
        m, n, k = cell.shape
        dtype = _torch_dtype(cell.dtype)
        x = torch.randn(m, k, device="cuda", dtype=dtype)
        y = torch.randn(k, n, device="cuda", dtype=dtype)
        inputs = {"x": x, "y": y}
        torch_fn = lambda: torch.matmul(x, y)  # noqa: E731
        torch_impl = "torch.matmul (cuBLAS)"
        reference_fn = torch_fn
        if want_helion:
            from examples.matmul_split_k import matmul_split_k

            kernels = [matmul_split_k]
            helion_fn = lambda: matmul_split_k(x, y)  # noqa: E731
    elif fam in ("fp8", "fp8rw"):
        m, n, k = cell.shape
        x = (torch.randn(m, k, device="cuda") * 0.4).to(torch.float8_e4m3fn)
        y = (torch.randn(k, n, device="cuda") * 0.4).to(torch.float8_e4m3fn)
        if cell.layout == "nt":
            y = y.T.contiguous().T
        inputs = {"x": x, "y": y}
        if fam == "fp8":
            from examples.fp8_gemm import reference_fp8_gemm_pytorch

            scale_a = torch.tensor(0.5, device="cuda")
            scale_b = torch.tensor(0.25, device="cuda")
            inputs.update(scale_a=scale_a, scale_b=scale_b)
            torch_fn = lambda: reference_fp8_gemm_pytorch(x, y, scale_a, scale_b)  # noqa: E731
            torch_impl = "torch._scaled_mm tensor-wise (cuBLASLt)"
            reference_fn = torch_fn
            if want_helion:
                from examples.fp8_gemm import fp8_gemm

                kernels = [fp8_gemm]
                helion_fn = lambda: fp8_gemm(x, y, scale_a, scale_b)  # noqa: E731
        else:
            from examples.fp8_matmul import reference_scaled_mm

            scale_a = (torch.rand(m, 1, device="cuda") + 0.5).float()
            scale_b = (torch.rand(1, n, device="cuda") + 0.5).float()
            scale_a2d = scale_a.reshape(m, 1).expand(m, n)
            scale_b1d = scale_b.reshape(n)
            inputs.update(scale_a=scale_a, scale_b=scale_b)
            torch_fn = lambda: reference_scaled_mm(x, y, scale_a, scale_b)  # noqa: E731
            torch_impl = "torch._scaled_mm row-wise (cuBLASLt)"
            reference_fn = torch_fn
            if want_helion:
                from examples.fp8_matmul import fp8_matmul

                kernels = [fp8_matmul]
                helion_fn = lambda: fp8_matmul(x, y, scale_a2d, scale_b1d)  # noqa: E731
    elif fam == "int4":
        m, n, k = cell.shape
        from examples.int4_gemm import _pack_int4_matrix
        from examples.int4_gemm import reference_matmul_bf16_int4

        A = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        B_unpacked = torch.randint(-8, 8, (k, n), dtype=torch.int8, device="cuda")
        B_packed = _pack_int4_matrix(B_unpacked)
        inputs = {"A": A, "B_packed": B_packed}
        reference_fn = lambda: reference_matmul_bf16_int4(A, B_packed)  # noqa: E731
        torch_fn = torch.compile(reference_fn)
        torch_impl = "torch.compile(int4 unpack + bf16 matmul)"
        if want_helion:
            from examples.int4_gemm import matmul_bf16_int4

            kernels = [matmul_bf16_int4]
            helion_fn = lambda: matmul_bf16_int4(A, B_packed)  # noqa: E731
    elif fam == "i16":
        m, n, k = cell.shape
        from examples.bf16xint16_gemm import reference_bf16xint16_pytorch

        transpose = cell.layout == "t"
        if transpose:
            x = torch.randint(
                -(2**15), 2**15 - 1, (m, k), device="cuda", dtype=torch.int16
            )
            w = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        else:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            w = torch.randint(
                -(2**15), 2**15 - 1, (k, n), device="cuda", dtype=torch.int16
            )
        inputs = {"x": x, "w": w}
        reference_fn = lambda: reference_bf16xint16_pytorch(x, w, transpose)  # noqa: E731
        torch_fn = torch.compile(reference_fn)
        torch_impl = "torch.compile(int16->bf16 cast + bf16 matmul)"
        if want_helion:
            from examples.bf16xint16_gemm import _bf16xint16_gemm
            from examples.bf16xint16_gemm import _int16xbf16_gemm
            from examples.bf16xint16_gemm import bf16xint16_gemm

            kernels = [_int16xbf16_gemm if transpose else _bf16xint16_gemm]
            helion_fn = lambda: bf16xint16_gemm(x, w, transpose)  # noqa: E731
    elif fam in ("grouped", "groupedp", "groupedv2"):
        total_m, n, k = cell.shape
        groups = int(cell.layout[1:])
        rows = _group_rows(total_m, groups, seed=groups)
        assert sum(rows) == total_m, rows
        A_packed = torch.randn(total_m, k, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        offsets = torch.tensor(
            [0, *list(_prefix(rows))], device="cuda", dtype=torch.int32
        )
        inputs = {"A_packed": A_packed, "B": B, "group_offsets": offsets, "rows": rows}
        # Every group shares B, so the grouped product is one GEMM over the
        # packed rows: the honest torch baseline is that GEMM.
        torch_fn = lambda: torch.matmul(A_packed, B)  # noqa: E731
        torch_impl = "torch.matmul over the packed rows (shared B)"
        reference_fn = torch_fn
        if want_helion:
            from examples.grouped_gemm import grouped_gemm_jagged
            from examples.grouped_gemm import grouped_gemm_jagged_persistent
            from examples.grouped_gemm import grouped_gemm_shared_weight

            kernel = {
                "groupedp": grouped_gemm_jagged_persistent,
                "groupedv2": grouped_gemm_shared_weight,
            }.get(fam, grouped_gemm_jagged)
            kernels = [kernel]
            helion_fn = lambda: kernel(A_packed, B, offsets)  # noqa: E731
    else:
        raise ValueError(f"unknown family {fam}")
    return Built(helion_fn, torch_fn, reference_fn, kernels, torch_impl, inputs)


def _prefix(rows: list[int]) -> list[int]:
    out = []
    total = 0
    for r in rows:
        total += r
        out.append(total)
    return out


# --------------------------------------------------------------------------
# Timing and correctness
# --------------------------------------------------------------------------


def _flatten(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [t for v in value for t in _flatten(v)]
    raise TypeError(type(value))


def check_correct(actual: Any, expected: Any, cell: Cell) -> dict[str, Any]:
    acts, exps = _flatten(actual), _flatten(expected)
    result: dict[str, Any] = {"correct": True}
    worst_abs = 0.0
    worst_mismatch = 0.0
    for a, e in zip(acts, exps, strict=True):
        if a.shape != e.shape:
            result["correct"] = False
            result["shape_error"] = f"{tuple(a.shape)} vs {tuple(e.shape)}"
            continue
        a32, e32 = a.float(), e.float()
        diff = (a32 - e32).abs()
        finite = torch.isfinite(a32)
        finite_ref = torch.isfinite(e32)
        allowed = cell.atol + cell.rtol * e32.abs()
        # A non-finite reference element is matched exactly (same inf, or
        # both NaN); a finite one by tolerance.
        same_nonfinite = (a32 == e32) | (torch.isnan(a32) & torch.isnan(e32))
        bad = torch.where(finite_ref, (diff > allowed) | ~finite, ~same_nonfinite)
        finite = finite & finite_ref
        mismatch = bad.float().mean().item()
        worst_abs = max(
            worst_abs, diff[finite].max().item() if finite.any() else math.inf
        )
        worst_mismatch = max(worst_mismatch, mismatch)
        if mismatch > cell.mismatch_budget:
            result["correct"] = False
    result.update(max_abs_err=worst_abs, mismatch_frac=worst_mismatch)
    return result


TIMER_KEYS = ("steady_ms", "bench_ms", "event_ms")
# The autotuner's final-selection steady window (``FinalSelectionConfig``:
# 500 + 500 ms flushed windows) after a thermal warmup of back-to-back
# launches that settles the B200 power limiter; the harness warms for 3 s
# where the final selection warms for 5 s, so a reading here sits a little
# earlier on the clock-settling curve than the autotuner's own.
STEADY_WARMUP_MS = 500
STEADY_REP_MS = 500
STEADY_THERMAL_MS = 3000


def steady_window_ms(fn: Callable[[], Any]) -> float:
    """One sustained-regime window: ``STEADY_WARMUP_MS`` of flushed calls then
    the median call time over ``STEADY_REP_MS`` more (CUDA-graph replays of
    [L2 flush; call])."""
    from helion.autotuner.benchmarking import do_bench as helion_do_bench

    return float(
        cast(
            "float",
            helion_do_bench(
                fn,
                warmup=STEADY_WARMUP_MS,
                rep=STEADY_REP_MS,
                return_mode="median",
                default_cudagraph=True,
                l2_flush_mode="call",
            ),
        )
    )


def steady_thermal_warmup(fns: list[Callable[[], Any]]) -> None:
    """Drive the GPU to its sustained clock state with back-to-back flushed
    calls of every function, one function after the other, each getting an
    equal share of ``STEADY_THERMAL_MS``."""
    from helion.autotuner.benchmarking import do_bench as helion_do_bench

    per_fn = max(100, STEADY_THERMAL_MS // max(1, len(fns)))
    for fn in fns:
        helion_do_bench(
            fn,
            warmup=per_fn,
            rep=1,
            default_cudagraph=True,
            l2_flush_mode="call",
        )


def time_fn(fn: Callable[[], Any], reps: int) -> dict[str, Any]:
    from triton.testing import do_bench

    from helion.autotuner.benchmarking import _make_cudagraph_replay
    from helion.autotuner.benchmarking import interleaved_bench

    result: dict[str, Any] = {}
    torch.cuda.synchronize()
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    warm = [
        float(do_bench(fn, warmup=10, rep=50, return_mode="median"))  # pyrefly: ignore[bad-argument-type]
        for _ in range(reps)
    ]
    result.update(event_ms=statistics.median(warm), event_runs_ms=warm)
    captured = False
    try:
        # Capture once first: a function that cannot be graph-captured fails
        # here, before any timing, and is recorded as ``bench_error``.
        _make_cudagraph_replay(fn)
        captured = True
        cold = [
            interleaved_bench(
                [fn], repeat=100, default_cudagraph=True, l2_flush_mode="call"
            )[0]
            for _ in range(reps)
        ]
        result.update(bench_ms=statistics.median(cold), bench_runs_ms=cold)
    except Exception as error:
        result["bench_error"] = f"{type(error).__name__}: {error}"
    if not captured:
        # ``do_bench`` would fall back to eager launches, a regime no other
        # record of the cell is timed in.
        result["steady_error"] = "skipped: the function cannot be graph-captured"
        return result
    try:
        steady_thermal_warmup([fn])
        steady = [steady_window_ms(fn) for _ in range(reps)]
        result.update(steady_ms=statistics.median(steady), steady_runs_ms=steady)
    except Exception as error:
        result["steady_error"] = f"{type(error).__name__}: {error}"
    return result


def preferred_ms(
    record: dict[str, Any] | None, timer: str | None = None
) -> float | None:
    """The record's time from ``timer`` (``steady_ms`` / ``bench_ms`` /
    ``event_ms``), or from the first timer of ``TIMER_KEYS`` it has."""
    if record is None or record.get("status") not in ("ok", "wrong"):
        return None
    keys = (timer,) if timer else TIMER_KEYS
    for key in keys:
        if record.get(key):
            return float(record[key])
    return None


def kernel_configs(kernels: list[Any]) -> list[dict[str, Any]]:
    out = []
    for kernel in kernels:
        out.append(
            {
                "kernel": f"{kernel.fn.__module__}.{kernel.fn.__name__}",
                "backend": kernel.settings.backend,
                "configs": [
                    json.loads(bound._config.to_json())
                    if bound._config is not None
                    else None
                    for bound in kernel._bound_kernels.values()
                ],
            }
        )
    return out


def pin_from_record(kernels: list[Any], record: dict[str, Any]) -> int:
    import helion

    pinned = 0
    by_name = {b["kernel"]: b for b in record.get("bound_kernels") or []}
    for kernel in kernels:
        name = f"{kernel.fn.__module__}.{kernel.fn.__name__}"
        bound = by_name.get(name)
        if bound is None:
            continue
        configs = [c for c in bound["configs"] if c]
        if not configs:
            continue
        kernel.configs = [helion.Config(**c) for c in configs]
        pinned += 1
    return pinned


def gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"name": torch.cuda.get_device_name(0)}
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,enforced.power.limit,temperature.gpu,clocks.sm",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        )
        lines = query.stdout.strip().splitlines()
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        info["visible"] = visible
        info["nvidia_smi"] = lines
        if visible.isdigit() and int(visible) < len(lines):
            info["nvidia_smi_visible"] = lines[int(visible)]
    except Exception as error:
        info["nvidia_smi_error"] = str(error)
    return info


# --------------------------------------------------------------------------
# worker: one cell x one provider in this process
# --------------------------------------------------------------------------


def configure_worker_env(args: argparse.Namespace) -> None:
    backend = args.provider if args.provider in ("cute", "triton") else "triton"
    os.environ["HELION_BACKEND"] = backend
    if args.pin_from is not None:
        os.environ.pop("HELION_FORCE_AUTOTUNE", None)
        os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "full")
    elif args.provider in ("cute", "triton"):
        os.environ["HELION_AUTOTUNE_EFFORT"] = args.effort
        os.environ["HELION_FORCE_AUTOTUNE"] = "1"
        os.environ.setdefault("HELION_AUTOTUNE_L2_FLUSH", "call")
        os.environ.setdefault("HELION_AUTOTUNE_FINAL_L2_FLUSH", "call")
        if args.seed is not None:
            os.environ["HELION_AUTOTUNE_RANDOM_SEED"] = str(args.seed)
    os.environ.setdefault("HELION_AUTOTUNE_PROGRESS_BAR", "0")


def worker(args: argparse.Namespace) -> None:
    if args.pin_from is not None:
        # A pinned config was validated when it was recorded; the harness
        # checks its result below. The autotuner's own check rejects a bf16
        # atomic split-K config on a one-element rounding flip.
        os.environ.setdefault("HELION_AUTOTUNE_ACCURACY_CHECK", "0")
    configure_worker_env(args)
    cell = cell_by_id(args.cell)
    torch.cuda.set_device(0)
    record: dict[str, Any] = {
        "cell": cell.id,
        "family": cell.family,
        "dtype": cell.dtype,
        "layout": cell.layout,
        "shape": list(cell.shape),
        "note": cell.note,
        "flops": cell.flops,
        "provider": args.provider,
        "effort": args.effort if args.pin_from is None else "pinned",
        "seed": args.seed,
        "autotune_l2_flush": os.environ.get("HELION_AUTOTUNE_L2_FLUSH"),
        "gpu": gpu_info(),
        "commit": _git_commit(),
        "status": "ok",
    }
    try:
        built = build(cell, args.provider)
        record["torch_impl"] = built.torch_impl
        if args.provider == "torch":
            fn = built.torch_fn
        else:
            assert built.helion_fn is not None
            fn = built.helion_fn
            if args.pin_from is not None:
                pinned_record = json.loads(Path(args.pin_from).read_text())
                pinned = pin_from_record(built.kernels, pinned_record)
                if not pinned:
                    raise RuntimeError(f"no kernel pinned from {args.pin_from}")
                record["pinned_from"] = str(args.pin_from)
        expected = built.reference_fn()
        torch.cuda.synchronize()
        started = time.monotonic()
        actual = fn()
        torch.cuda.synchronize()
        record["first_call_seconds"] = time.monotonic() - started
        record.update(check_correct(actual, expected, cell))
        if args.provider != "torch":
            record["bound_kernels"] = kernel_configs(built.kernels)
            for bound in record["bound_kernels"]:
                for config in bound["configs"]:
                    print(
                        f"config {bound['kernel']}: {json.dumps(config)}",
                        file=sys.stderr,
                    )
        record.update(time_fn(fn, args.reps))
        if not record["correct"]:
            record["status"] = "wrong"
        ms = preferred_ms(record)
        if ms:
            record["tflops"] = cell.flops / ms / 1e9
    except Exception as error:
        record["status"] = "error"
        record["error"] = f"{type(error).__name__}: {error}"
        record["traceback"] = traceback.format_exc()
    record["gpu_after"] = gpu_info()
    summary = {
        k: record.get(k)
        for k in (
            "status",
            "steady_ms",
            "bench_ms",
            "event_ms",
            "tflops",
            "max_abs_err",
            "first_call_seconds",
            "error",
        )
        if record.get(k) is not None
    }
    print(f"RESULT {cell.id} {args.provider} {json.dumps(summary)}", flush=True)
    if args.json is not None:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(record, indent=2, default=str) + "\n")
    if record["status"] == "error":
        print(record["traceback"], file=sys.stderr)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
            timeout=20,
        ).stdout.strip()
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------
# abab: interleaved cute-vs-torch rounds in one process (pinned config)
# --------------------------------------------------------------------------


def abab(args: argparse.Namespace) -> None:
    os.environ["HELION_BACKEND"] = "cute"
    os.environ.pop("HELION_FORCE_AUTOTUNE", None)
    os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "full")
    from helion.autotuner.benchmarking import interleaved_bench

    cell = cell_by_id(args.cell)
    torch.cuda.set_device(0)
    built = build(cell, "cute")
    assert built.helion_fn is not None
    pinned_record = json.loads(Path(args.pin_from).read_text())
    if not pin_from_record(built.kernels, pinned_record):
        raise SystemExit(f"no kernel pinned from {args.pin_from}")
    expected = built.reference_fn()
    actual = built.helion_fn()
    correctness = check_correct(actual, expected, cell)
    fns = [built.helion_fn, built.torch_fn]
    for fn in fns:
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    rounds: list[list[float]] = []
    for _ in range(args.rounds):
        rounds.append(
            interleaved_bench(
                fns, repeat=100, default_cudagraph=True, l2_flush_mode="call"
            )
        )
    cute_ms = statistics.median(r[0] for r in rounds)
    torch_ms = statistics.median(r[1] for r in rounds)
    # Sustained regime: alternate one steady window of each implementation
    # per round, flipping the order every round so clock drift cancels.
    steady_thermal_warmup(fns)
    steady_rounds: list[list[float]] = []
    for index in range(args.rounds):
        order = [0, 1] if index % 2 == 0 else [1, 0]
        readings = [0.0, 0.0]
        for which in order:
            readings[which] = steady_window_ms(fns[which])
        steady_rounds.append(readings)
    steady_cute_ms = statistics.median(r[0] for r in steady_rounds)
    steady_torch_ms = statistics.median(r[1] for r in steady_rounds)
    record = {
        "cell": cell.id,
        "mode": "abab",
        "pinned_from": str(args.pin_from),
        "rounds": rounds,
        "cute_ms": cute_ms,
        "torch_ms": torch_ms,
        "ratio": torch_ms / cute_ms,
        "paired_ratios": [r[1] / r[0] for r in rounds],
        "steady_rounds": steady_rounds,
        "steady_cute_ms": steady_cute_ms,
        "steady_torch_ms": steady_torch_ms,
        "steady_ratio": steady_torch_ms / steady_cute_ms,
        "steady_paired_ratios": [r[1] / r[0] for r in steady_rounds],
        "gpu": gpu_info(),
        "commit": _git_commit(),
        **correctness,
    }
    print(
        f"ABAB {cell.id} burst: cute {cute_ms:.4f} ms torch {torch_ms:.4f} ms ratio {record['ratio']:.4f}"
        f" | steady: cute {steady_cute_ms:.4f} ms torch {steady_torch_ms:.4f} ms ratio {record['steady_ratio']:.4f}"
    )
    if args.json is not None:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(record, indent=2, default=str) + "\n")


# --------------------------------------------------------------------------
# run: lanes of subprocesses, one GPU each
# --------------------------------------------------------------------------


def worker_environment(
    gpu: str, provider: str, cache: Path, seed: int, log: Path
) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("HELION_")}
    env.update(
        CUDA_VISIBLE_DEVICES=gpu,
        HELION_BACKEND=provider if provider in ("cute", "triton") else "triton",
        HELION_AUTOTUNE_RANDOM_SEED=str(seed),
        HELION_AUTOTUNE_L2_FLUSH="call",
        HELION_AUTOTUNE_FINAL_L2_FLUSH="call",
        HELION_AUTOTUNE_PROGRESS_BAR="0",
        HELION_AUTOTUNE_LOG=str(log),
        HELION_CACHE_DIR=str(cache / "helion"),
        TRITON_CACHE_DIR=str(cache / "triton"),
        TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
        CUTE_DSL_CACHE_DIR=str(cache / "cute"),
        CUDA_CACHE_PATH=str(cache / "cuda"),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
        PYTHONPATH=os.pathsep.join([str(REPO_ROOT), env.get("PYTHONPATH", "")]),
    )
    return env


def foreign_processes(gpu: str) -> list[str]:
    try:
        uuid = subprocess.run(
            ["nvidia-smi", "-i", gpu, "--query-gpu=uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        ).stdout.strip()
        apps = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        ).stdout
    except Exception:
        return []
    return [line for line in apps.splitlines() if line.startswith(uuid)]


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def record_path(out: Path, cell: Cell, provider: str) -> Path:
    return out / f"{cell.id}__{provider}.json"


def run(args: argparse.Namespace) -> None:
    out = Path(args.out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "pids").mkdir(exist_ok=True)
    (out / "claims").mkdir(exist_ok=True)
    # A claim file hands a job to one lane of one driver; claims of drivers
    # that are gone (a killed or crashed run over the same directory) would
    # otherwise keep every retried job skipped.
    for claim in (out / "claims").iterdir():
        try:
            owner = int(claim.read_text().split()[0])
        except (OSError, ValueError, IndexError):
            continue
        if not _pid_alive(owner):
            claim.unlink(missing_ok=True)
    (out / "pids" / f"driver-{os.getpid()}.pid").write_text(f"{os.getpid()}\n")
    cells = select_cells(args.cells)
    providers = args.providers.split(",")
    jobs: list[tuple[Cell, str]] = []
    for cell in cells:
        for provider in providers:
            path = record_path(out, cell, provider)
            if path.exists() and not args.redo:
                status = json.loads(path.read_text()).get("status")
                if (
                    status == "ok"
                    or (status == "wrong" and not args.retry_wrong)
                    or (status == "error" and not args.retry_errors)
                ):
                    continue
            jobs.append((cell, provider))
    # Big cells first so the lanes drain evenly.
    jobs.sort(key=lambda j: -j[0].flops)
    print(f"{len(jobs)} jobs over GPUs {args.gpus}", flush=True)
    lock = threading.Lock()
    log_file = open(out / "driver.log", "a")

    def log(message: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        with lock:
            print(line, flush=True)
            log_file.write(line + "\n")
            log_file.flush()

    def lane(gpu: str) -> None:
        while True:
            with lock:
                if (
                    not jobs
                    or (out / "stop").exists()
                    or (out / f"stop-gpu{gpu}").exists()
                ):
                    return
                cell, provider = jobs.pop(0)
            if args.pin_dir is not None and provider != "torch":
                pinned_path = record_path(Path(args.pin_dir), cell, provider)
                if not pinned_path.exists():
                    log(
                        f"SKIP  gpu{gpu} {cell.id}__{provider}: no record to pin in {args.pin_dir}"
                    )
                    continue
            # Several drivers may share one output directory (lanes added
            # later): a claim file hands each job to exactly one lane.
            try:
                claim = os.open(
                    out / "claims" / f"{cell.id}__{provider}",
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
            except FileExistsError:
                log(f"SKIP  gpu{gpu} {cell.id}__{provider}: claimed by another lane")
                continue
            os.write(claim, f"{os.getpid()} gpu{gpu}\n".encode())
            os.close(claim)
            foreign_at_start = foreign_processes(gpu)
            while foreign_at_start and not args.share_gpus:
                log(f"gpu{gpu} busy with a foreign process; waiting")
                time.sleep(60)
                foreign_at_start = foreign_processes(gpu)
            seed = secrets.randbits(31)
            name = f"{cell.id}__{provider}"
            with tempfile.TemporaryDirectory(prefix="matmul-sweep-") as cache:
                env = worker_environment(
                    gpu,
                    provider,
                    Path(cache),
                    seed,
                    out / "logs" / f"{name}.autotune.log",
                )
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "worker",
                    "--cell",
                    cell.id,
                    "--provider",
                    provider,
                    "--effort",
                    args.effort,
                    "--seed",
                    str(seed),
                    "--reps",
                    str(args.reps),
                    "--json",
                    str(record_path(out, cell, provider)),
                ]
                if args.pin_dir is not None and provider != "torch":
                    command.extend(
                        [
                            "--pin-from",
                            str(record_path(Path(args.pin_dir), cell, provider)),
                        ]
                    )
                # A rerun (``--redo``, a retry, a timeout or crash requeue)
                # must not inherit the old record: a worker that dies before
                # writing its JSON would otherwise leave the old status in
                # place, re-annotated as if it were this run's.
                record_path(out, cell, provider).unlink(missing_ok=True)
                log(f"START gpu{gpu} {name} seed={seed}")
                started = time.monotonic()
                with open(out / "logs" / f"{name}.log", "w") as handle:
                    proc = subprocess.Popen(
                        command,
                        env=env,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        cwd=REPO_ROOT,
                        # The worker leads its own session so a timeout can
                        # take the autotuner's spawned children with it
                        # (their CUDA contexts would otherwise look like
                        # foreign tenants to the lanes).
                        start_new_session=True,
                    )
                    (out / "pids" / f"gpu{gpu}.pid").write_text(f"{proc.pid}\n")
                    try:
                        code = proc.wait(timeout=args.timeout)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                        code = "timeout"
                        record_path(out, cell, provider).write_text(
                            json.dumps(
                                {
                                    "cell": cell.id,
                                    "provider": provider,
                                    "status": "timeout",
                                    "timeout_seconds": args.timeout,
                                    "effort": args.effort,
                                },
                                indent=2,
                            )
                            + "\n"
                        )
                    else:
                        # A worker that crashed can leave autotuner children
                        # behind; their CUDA contexts would look like foreign
                        # tenants to every lane. The group is gone already
                        # when the worker exited cleanly.
                        with contextlib.suppress(ProcessLookupError):
                            os.killpg(proc.pid, signal.SIGKILL)
                elapsed = time.monotonic() - started
                path = record_path(out, cell, provider)
                status = "missing"
                if path.exists():
                    # Annotate the record with the GPU's other tenants: a
                    # reading taken beside a foreign process is suspect.
                    record = json.loads(path.read_text())
                    record["foreign_at_start"] = foreign_at_start
                    record["foreign_at_end"] = foreign_processes(gpu)
                    record["driver_gpu"] = gpu
                    path.write_text(json.dumps(record, indent=2, default=str) + "\n")
                    status = record.get("status")
                if not path.exists():
                    path.write_text(
                        json.dumps(
                            {
                                "cell": cell.id,
                                "provider": provider,
                                "status": "crash",
                                "exit_code": code,
                                "effort": args.effort,
                            },
                            indent=2,
                        )
                        + "\n"
                    )
                    status = "crash"
                log(
                    f"DONE  gpu{gpu} {name} status={status} exit={code} {elapsed / 60:.1f} min"
                )

    threads = [
        threading.Thread(target=lane, args=(gpu,), name=f"gpu{gpu}")
        for gpu in args.gpus.split(",")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    log("ALL LANES DONE")
    log_file.close()


# --------------------------------------------------------------------------
# summarize
# --------------------------------------------------------------------------


def load_records(out: Path) -> dict[tuple[str, str], dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(out.glob("*__*.json")):
        record = json.loads(path.read_text())
        record["_path"] = str(path)
        records[(record["cell"], record["provider"])] = record
    return records


def _fmt_ms(value: float | None) -> str:
    if value is None:
        return "-"
    if value < 0.1:
        return f"{value * 1000:.1f}us"
    return f"{value:.3f}"


def _fmt_ratio(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _ratio(
    baseline: dict[str, Any] | None,
    cute: dict[str, Any] | None,
    timer: str | None = None,
) -> tuple[float | None, str | None]:
    """baseline / cute time from one timer, or None.

    Both records must have run correctly (a wrong result is not a timing),
    and both must come from the same regime: ``timer`` when given, else the
    first of ``TIMER_KEYS`` (sustained, cold burst, warm) both records have.
    """
    if baseline is None or cute is None:
        return None, None
    if baseline.get("status") != "ok" or cute.get("status") != "ok":
        return None, None
    for key in (timer,) if timer else TIMER_KEYS:
        if baseline.get(key) and cute.get(key):
            return baseline[key] / cute[key], key
    return None, None


def summarize(args: argparse.Namespace) -> None:
    out = Path(args.out)
    records = load_records(out)
    if args.baseline_out:
        for key, record in load_records(Path(args.baseline_out)).items():
            records.setdefault(key, record)
    cells = select_cells(args.cells)
    rows: list[dict[str, Any]] = []
    for cell in cells:
        torch_rec = records.get((cell.id, "torch"))
        cute_rec = records.get((cell.id, "cute"))
        triton_rec = records.get((cell.id, "triton"))
        ratio, timer = _ratio(torch_rec, cute_rec, args.timer)
        # Every column of a row reads the ratio's timer, so a row never mixes
        # regimes (an old record lacking that timer shows no time).
        row_timer = timer or args.timer or TIMER_KEYS[0]
        ratio_triton, _ = _ratio(triton_rec, cute_rec, row_timer)
        torch_ms, cute_ms, triton_ms = (
            preferred_ms(torch_rec, row_timer),
            preferred_ms(cute_rec, row_timer),
            preferred_ms(triton_rec, row_timer),
        )
        status = (cute_rec or {}).get("status", "missing")
        if status == "ok" and ratio is not None:
            if ratio < 0.9:
                status = "SLOW"
            elif ratio < 0.99:
                status = "behind"
            elif ratio > 1.1:
                status = "FAST"
            else:
                status = "par"
        rows.append(
            {
                "cell": cell.id,
                "note": cell.note,
                "torch_ms": torch_ms,
                "triton_ms": triton_ms,
                "cute_ms": cute_ms,
                "ratio": ratio,
                "ratio_timer": timer,
                "ratio_vs_triton": ratio_triton,
                "cute_tflops": (cell.flops / cute_ms / 1e9) if cute_ms else None,
                "torch_tflops": (cell.flops / torch_ms / 1e9) if torch_ms else None,
                "status": status,
                "cute_status": (cute_rec or {}).get("status", "missing"),
                "triton_status": (triton_rec or {}).get("status", "missing"),
                "torch_status": (torch_rec or {}).get("status", "missing"),
                "cute_error": ((cute_rec or {}).get("error") or "")[:160],
                "cute_max_abs_err": (cute_rec or {}).get("max_abs_err"),
                # Every bound kernel's configs (a two-launch cell such as
                # ``mmln2`` records the matmul's and the layernorm's).
                "cute_config": [
                    {"kernel": bound.get("kernel"), "configs": bound.get("configs")}
                    for bound in (cute_rec or {}).get("bound_kernels") or []
                ],
                "torch_impl": (torch_rec or {}).get("torch_impl"),
                "shared_gpu": any(
                    (rec or {}).get("foreign_at_start")
                    or (rec or {}).get("foreign_at_end")
                    for rec in (torch_rec, cute_rec, triton_rec)
                ),
            }
        )
    ratios: list[float] = [r["ratio"] for r in rows if r["ratio"]]
    geomean = (
        math.exp(sum(math.log(r) for r in ratios) / len(ratios)) if ratios else None
    )
    lines = [
        f"# matmul sweep summary: {out}",
        "",
        f"{len(rows)} cells; {len(ratios)} with cute+torch readings; geomean torch/cute = {_fmt_ratio(geomean)}",
        "",
        "| cell | note | torch ms | triton ms | cute ms | torch/cute | triton/cute | cute TF/s | status |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['cell']} | {r['note']} | {_fmt_ms(r['torch_ms'])} | {_fmt_ms(r['triton_ms'])} | "
            f"{_fmt_ms(r['cute_ms'])} | {_fmt_ratio(r['ratio'])} | {_fmt_ratio(r['ratio_vs_triton'])} | "
            f"{(r['cute_tflops'] or 0):.0f} | {r['status']}{' (shared GPU)' if r['shared_gpu'] else ''}"
            f"{' ' + r['cute_error'] if r['cute_error'] else ''} |"
        )
    text = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(text)
    (out / "summary.json").write_text(
        json.dumps({"geomean": geomean, "rows": rows}, indent=2, default=str) + "\n"
    )
    print(text)
    counts: dict[str, int] = {}
    for r in rows:
        status = str(r["status"])
        counts[status] = counts.get(status, 0) + 1
    print("status counts:", json.dumps(counts))


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list")
    p_list.add_argument("--cells", default=None, help="regex over cell ids")

    p_worker = sub.add_parser("worker")
    p_worker.add_argument("--cell", required=True)
    p_worker.add_argument("--provider", required=True, choices=PROVIDERS)
    p_worker.add_argument("--effort", default="full")
    p_worker.add_argument("--seed", type=int, default=None)
    p_worker.add_argument("--reps", type=int, default=5)
    p_worker.add_argument("--pin-from", type=Path, default=None)
    p_worker.add_argument("--json", type=Path, default=None)

    p_run = sub.add_parser("run")
    p_run.add_argument("--out", required=True)
    p_run.add_argument("--gpus", required=True, help="comma-separated physical GPU ids")
    p_run.add_argument("--providers", default="cute,triton,torch")
    p_run.add_argument("--effort", default="quick")
    p_run.add_argument("--cells", default=None, help="regex over cell ids")
    p_run.add_argument("--reps", type=int, default=5)
    p_run.add_argument("--timeout", type=int, default=5400)
    p_run.add_argument("--redo", action="store_true")
    p_run.add_argument("--retry-errors", action="store_true")
    p_run.add_argument("--retry-wrong", action="store_true")
    p_run.add_argument(
        "--pin-dir",
        type=Path,
        default=None,
        help="re-time the cute/triton configs recorded in this directory instead of autotuning",
    )
    p_run.add_argument(
        "--share-gpus",
        action="store_true",
        help="Start jobs beside foreign GPU processes instead of waiting "
        "(records are annotated with foreign_at_start / foreign_at_end)",
    )

    p_sum = sub.add_parser("summarize")
    p_sum.add_argument(
        "--baseline-out",
        default=None,
        help="directory whose records fill providers missing from --out "
        "(a cute-only rerun summarized against the landscape pass's baselines)",
    )
    p_sum.add_argument("--out", required=True)
    p_sum.add_argument("--cells", default=None)
    p_sum.add_argument(
        "--timer",
        default=None,
        choices=TIMER_KEYS,
        help="time both providers by this regime (default: the first regime both records have)",
    )

    p_abab = sub.add_parser("abab")
    p_abab.add_argument("--cell", required=True)
    p_abab.add_argument("--pin-from", type=Path, required=True)
    p_abab.add_argument("--rounds", type=int, default=5)
    p_abab.add_argument("--json", type=Path, default=None)

    args = parser.parse_args()
    if args.command == "list":
        cells = select_cells(args.cells)
        for cell in cells:
            print(f"{cell.id:48s} {cell.flops / 1e9:10.1f} GFLOP  {cell.note}")
        print(f"{len(cells)} cells")
    elif args.command == "worker":
        worker(args)
    elif args.command == "run":
        run(args)
    elif args.command == "summarize":
        summarize(args)
    elif args.command == "abab":
        abab(args)


if __name__ == "__main__":
    main()
