"""Shared helpers for the jagged_sum source variants."""
from __future__ import annotations

import traceback
from typing import Callable

import torch

from helion._testing import DEVICE


# Round-2 size sweep. We want to see where the Helion Pallas lowering breaks
# as L * M (the VMEM scratch the existing lowering allocates for the whole
# input) grows. Each tier has predictable scratch size at fp32:
#   small  : ~128 KB     baseline (known to compile)
#   medium : ~1   MB     should still fit
#   large  : ~8   MB     close to single-core VMEM on v4
#   huge   : ~64  MB     expected OOM / scratch error
SIZES: list[tuple[str, dict]] = [
    ("small",  dict(B=8,   max_seqlen=64)),
    ("medium", dict(B=16,  max_seqlen=256)),
    ("large",  dict(B=32,  max_seqlen=1024)),
    ("huge",   dict(B=64,  max_seqlen=4096)),
]


def make_data(B: int = 8, M: int = 128, max_seqlen: int = 64,
              dtype: torch.dtype = torch.float32, seed: int = 0):
    """Generate a jagged input. Offsets are int32 — Pallas/TPU rejects int64."""
    torch.manual_seed(seed)
    seq_lengths = torch.randint(1, max_seqlen + 1, (B,))
    x_offsets = torch.cat(
        [torch.zeros(1, dtype=seq_lengths.dtype), torch.cumsum(seq_lengths, dim=0)]
    ).to(torch.int32).to(DEVICE)
    nnz = int(x_offsets[-1])
    x_data = torch.randn(nnz, M, dtype=dtype).to(DEVICE)
    return x_data, x_offsets


def reference(x_data: torch.Tensor, x_offsets: torch.Tensor) -> torch.Tensor:
    num_rows = x_offsets.numel() - 1
    M = x_data.size(1)
    out = torch.zeros((num_rows, M), dtype=x_data.dtype, device=x_data.device)
    for i in range(num_rows):
        s = int(x_offsets[i])
        e = int(x_offsets[i + 1])
        if e > s:
            out[i, :] = x_data[s:e, :].sum(dim=0)
    return out


def banner(title: str) -> None:
    print("=" * 78, flush=True)
    print(f"=== {title}", flush=True)
    print("=" * 78, flush=True)


def size_banner(name: str, params: dict) -> None:
    print("-" * 78, flush=True)
    print(f"--- size '{name}': {params}", flush=True)
    print("-" * 78, flush=True)


def report_inputs(x_data: torch.Tensor, x_offsets: torch.Tensor) -> None:
    nnz_total = int(x_offsets[-1])
    bytes_full_scratch = nnz_total * x_data.size(1) * x_data.element_size()
    print(
        f"INPUT  x_data.shape={tuple(x_data.shape)} dtype={x_data.dtype} "
        f"device={x_data.device}",
        flush=True,
    )
    print(
        f"INPUT  x_offsets.shape={tuple(x_offsets.shape)} dtype={x_offsets.dtype}",
        flush=True,
    )
    # Don't dump full offset list at large sizes — keep it readable.
    head_tail = x_offsets.tolist() if x_offsets.numel() <= 16 else (
        x_offsets[:8].tolist() + ["..."] + x_offsets[-2:].tolist()
    )
    print(f"INPUT  x_offsets head/tail = {head_tail}", flush=True)
    print(
        f"INPUT  L=nnz_total={nnz_total}; if backend allocates full-input "
        f"VMEM scratch, that's ~{bytes_full_scratch/1024:.1f} KiB "
        f"(~{bytes_full_scratch/(1024*1024):.2f} MiB)",
        flush=True,
    )


def check_close(out: torch.Tensor, ref: torch.Tensor) -> None:
    diff = (out - ref).abs().max().item()
    print(f"RESULT out.shape={tuple(out.shape)}  max|diff|={diff:.6e}", flush=True)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
    print("RESULT correctness: OK", flush=True)


def run_all_sizes(
    variant_name: str,
    kernel_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    sizes: list[tuple[str, dict]] = SIZES,
) -> int:
    """Run the variant across multiple input sizes. Returns 0 if all sizes
    succeeded; nonzero if any size failed (we still try every size so we can
    map the breakage)."""
    banner(variant_name)
    overall_rc = 0
    for name, params in sizes:
        size_banner(name, params)
        try:
            x_data, x_offsets = make_data(**params)
        except Exception:
            print(f"SIZE '{name}': INPUT-GEN FAILED — full traceback:", flush=True)
            traceback.print_exc()
            overall_rc = max(overall_rc, 3)
            continue
        report_inputs(x_data, x_offsets)
        try:
            out = kernel_fn(x_data, x_offsets)
        except Exception:
            print(f"SIZE '{name}': COMPILE/RUN FAILED — full traceback:", flush=True)
            traceback.print_exc()
            overall_rc = max(overall_rc, 1)
            continue
        try:
            ref = reference(x_data, x_offsets)
            check_close(out, ref)
        except AssertionError:
            print(f"SIZE '{name}': CORRECTNESS FAILED — full traceback:", flush=True)
            traceback.print_exc()
            overall_rc = max(overall_rc, 2)
            continue
        print(f"SIZE '{name}': PASS", flush=True)
    return overall_rc
