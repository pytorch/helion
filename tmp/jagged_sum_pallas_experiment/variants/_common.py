"""Shared helpers for the three jagged_sum source variants."""
from __future__ import annotations

import torch

from helion._testing import DEVICE


def make_data(B: int = 8, M: int = 128, max_seqlen: int = 64,
              dtype: torch.dtype = torch.float32, seed: int = 0):
    torch.manual_seed(seed)
    seq_lengths = torch.randint(1, max_seqlen + 1, (B,))
    x_offsets = torch.cat(
        [torch.zeros(1, dtype=torch.long), torch.cumsum(seq_lengths, dim=0)]
    ).to(DEVICE)
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


def report_inputs(x_data: torch.Tensor, x_offsets: torch.Tensor) -> None:
    print(
        f"INPUT  x_data.shape={tuple(x_data.shape)} dtype={x_data.dtype} "
        f"device={x_data.device}",
        flush=True,
    )
    print(
        f"INPUT  x_offsets.shape={tuple(x_offsets.shape)} dtype={x_offsets.dtype} "
        f"value={x_offsets.tolist()}",
        flush=True,
    )
    seq_lens = (x_offsets[1:] - x_offsets[:-1]).tolist()
    print(f"INPUT  per-batch nnz = {seq_lens}", flush=True)


def check_close(out: torch.Tensor, ref: torch.Tensor) -> None:
    diff = (out - ref).abs().max().item()
    print(f"RESULT out.shape={tuple(out.shape)}  max|diff|={diff:.6e}", flush=True)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
    print("RESULT correctness: OK", flush=True)
