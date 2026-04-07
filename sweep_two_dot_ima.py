"""Systematic sweep of (K, K_bs) combinations for Triton two-dot IMA bug.

Each combo runs in a separate subprocess to avoid GPU state corruption.

Usage:
    python sweep_two_dot_ima.py

Known bug: Triton has a shared-memory bug when a kernel has 2+ tl.dot ops
in a batch loop (Batch >= 2). K_bs in {16, 32} triggers either IMA or
silent corruption. K_bs=8 and K_bs>=64 are safe.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile


TWO_DOT_KERNEL = """\
import torch
from unittest.mock import patch
import helion
import helion.language as hl
from helion import _compat


@helion.kernel(static_shapes=True)
def two_dot_bwd(
    x: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    Batch, N, M = x.size()
    K = A.size(0)
    out1 = torch.empty((Batch, N, K), device=x.device, dtype=x.dtype)
    out2 = torch.empty((Batch, M, K), device=x.device, dtype=x.dtype)
    for tile_b in hl.tile(Batch, block_size=1):
        for tile_n, tile_k in hl.tile([N, K]):
            acc = hl.zeros([tile_n, tile_k], dtype=torch.float32)
            for tile_m in hl.tile(M):
                acc = torch.addmm(acc, x[tile_b.begin, tile_n, tile_m], A[tile_k, tile_m].t())
            out1[tile_b.begin, tile_n, tile_k] = acc.to(x.dtype)
        for tile_m2, tile_k2 in hl.tile([M, K]):
            acc2 = hl.zeros([tile_m2, tile_k2], dtype=torch.float32)
            for tile_n2 in hl.tile(N):
                acc2 = torch.addmm(acc2, x[tile_b.begin, tile_n2, tile_m2].t(), B[tile_n2, tile_k2])
            out2[tile_b.begin, tile_m2, tile_k2] = acc2.to(x.dtype)
    return out1, out2


K, N, M = {K}, {N}, {M}
K_bs = {K_bs}
x = torch.randn(2, N, M, device="cuda", dtype=torch.bfloat16)
A = torch.randn(K, M, device="cuda", dtype=torch.bfloat16)
B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
args = (x, A, B)

config = helion.Config(
    block_sizes=[64, K_bs, 64, 64, K_bs, 64],
    indexing="pointer",
    num_stages=1,
    num_warps=4,
)

with patch.object(_compat, "_supports_tensor_descriptor", lambda: False):
    bound = two_dot_bwd.bind(args)
    compiled = bound.compile_config(config)
    for _ in range(10):
        torch.cuda.synchronize()
        out1, out2 = compiled(*args)
        torch.cuda.synchronize()

ref_out1 = torch.bmm(x.float(), A.float().t().unsqueeze(0).expand(2, -1, -1)).to(x.dtype)
ref_out2 = torch.bmm(x.float().transpose(-2, -1), B.float().unsqueeze(0).expand(2, -1, -1)).to(x.dtype)
torch.testing.assert_close(out1, ref_out1, atol=1e-1, rtol=1e-1)
torch.testing.assert_close(out2, ref_out2, atol=1e-1, rtol=1e-1)
print("OK")
"""

SINGLE_DOT_KERNEL = """\
import torch
from unittest.mock import patch
import helion
import helion.language as hl
from helion import _compat


@helion.kernel(static_shapes=True)
def single_dot(
    x: torch.Tensor,
    A: torch.Tensor,
) -> torch.Tensor:
    Batch, N, M = x.size()
    K = A.size(0)
    out1 = torch.empty((Batch, N, K), device=x.device, dtype=x.dtype)
    for tile_b in hl.tile(Batch, block_size=1):
        for tile_n, tile_k in hl.tile([N, K]):
            acc = hl.zeros([tile_n, tile_k], dtype=torch.float32)
            for tile_m in hl.tile(M):
                acc = torch.addmm(acc, x[tile_b.begin, tile_n, tile_m], A[tile_k, tile_m].t())
            out1[tile_b.begin, tile_n, tile_k] = acc.to(x.dtype)
    return out1


K, N, M = {K}, {N}, {M}
K_bs = {K_bs}
x = torch.randn(2, N, M, device="cuda", dtype=torch.bfloat16)
A = torch.randn(K, M, device="cuda", dtype=torch.bfloat16)
args = (x, A)

config = helion.Config(
    block_sizes=[64, K_bs, 64],
    indexing="pointer",
    num_stages=1,
    num_warps=4,
)

with patch.object(_compat, "_supports_tensor_descriptor", lambda: False):
    bound = single_dot.bind(args)
    compiled = bound.compile_config(config)
    for _ in range(10):
        torch.cuda.synchronize()
        out1 = compiled(*args)
        torch.cuda.synchronize()

ref_out1 = torch.bmm(x.float(), A.float().t().unsqueeze(0).expand(2, -1, -1)).to(x.dtype)
torch.testing.assert_close(out1, ref_out1, atol=1e-1, rtol=1e-1)
print("OK")
"""


def run_test(K: int, K_bs: int, num_dots: int = 2, N: int = 192, M: int = 128) -> str:
    """Run a test in a subprocess. Returns 'OK', 'IMA', 'WRONG', or error string."""
    template = TWO_DOT_KERNEL if num_dots == 2 else SINGLE_DOT_KERNEL
    script = template.format(K=K, N=N, M=M, K_bs=K_bs)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", dir=os.path.dirname(__file__) or ".", delete=False
    ) as f:
        f.write(script)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.lower()
        if result.returncode == 0 and "OK" in stdout:
            return "OK"
        elif any(s in stderr for s in ["illegal memory access", "misaligned address", "unspecified launch failure"]):
            return "IMA"
        elif "mismatched elements" in stderr:
            return "WRONG"
        else:
            last = result.stderr.strip().split("\n")[-1]
            return f"ERR({last[:60]})"
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    finally:
        os.unlink(tmp_path)


def print_table(title: str, K_values: list[int], K_bs_values: list[int], num_dots: int) -> None:
    """Run a sweep and print results as a table."""
    header = f"{'K':>6} |" + "|".join(f" bs={bs:>3}  " for bs in K_bs_values) + "|"
    sep = "-" * len(header)

    print(sep)
    print(title)
    print(sep)
    print(header)
    print(sep)

    for K in K_values:
        row = f"{K:>6} |"
        for K_bs in K_bs_values:
            if K_bs > K * 4:
                cell = "  skip  "
            else:
                r = run_test(K, K_bs, num_dots=num_dots)
                if r == "OK":
                    cell = "   OK   "
                elif r == "IMA":
                    cell = " *IMA*  "
                elif r == "WRONG":
                    cell = " WRONG  "
                elif r == "TIMEOUT":
                    cell = " TOUT   "
                else:
                    cell = " ERR    "
                    print(f"    [{K},{K_bs}]: {r}", file=sys.stderr)
            row += cell + "|"
        print(row, flush=True)

    print(sep)


def main() -> None:
    import torch

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    print(f"GPU: {gpu_name}")
    print(f"Python: {sys.version.split()[0]}")
    try:
        import triton
        print(f"Triton: {triton.__version__}")
    except Exception:
        print("Triton: unknown")
    print()

    # ── Part 1: Single-dot control ─────────────────────────────────────
    print_table(
        "PART 1: Single-dot kernel (control) — should all be OK",
        K_values=[48, 128],
        K_bs_values=[8, 16, 32, 64, 128],
        num_dots=1,
    )
    print()

    # ── Part 2: Two-dot full sweep ─────────────────────────────────────
    print_table(
        "PART 2: Two-dot kernel — full K x K_bs sweep",
        K_values=[16, 24, 32, 48, 64, 80, 96, 128, 256, 512, 1024],
        K_bs_values=[8, 16, 32, 64, 128, 256],
        num_dots=2,
    )
    print()

    # ── Part 3: Determinism check ──────────────────────────────────────
    print("=" * 60)
    print("PART 3: Determinism check — repeat 5x for known-bad combos")
    print("=" * 60)
    for K, K_bs in [(48, 16), (48, 32), (128, 16), (128, 32)]:
        results = [run_test(K, K_bs, num_dots=2) for _ in range(5)]
        print(f"  K={K:>4} K_bs={K_bs:>3}: {results}")
    print()

    print("Done.")


if __name__ == "__main__":
    main()

