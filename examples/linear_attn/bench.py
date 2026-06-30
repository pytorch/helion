"""
Linear Attention — local benchmark vs FLA
=========================================

Runs Helion's chunk_linear_attn against FLA on the 6 production shapes from
flash-linear-attention/benchmarks/ops/registry.py:184. Reports forward and
forward+backward timings on the local H100.

Not a CI artifact — just a dev tool for tracking perf as we tune.

Usage:
    HELION_AUTOTUNE_EFFORT=none python -m examples.linear_attn.bench
    HELION_AUTOTUNE_EFFORT=quick python -m examples.linear_attn.bench --fwd-only
"""

from __future__ import annotations

import argparse
import math

import torch
from triton.testing import do_bench

from .chunk import chunk_linear_attn
from .chunk import fla_chunk_linear_attn
from helion._testing import DEVICE

# (name, B, T, H, D)  — D is both query/key dim and value dim (D=DV) per FLA registry.
# Shapes from flash-linear-attention/benchmarks/ops/registry.py:184.
SHAPES = [
    ("B1_T8192_H96_D128", 1, 8192, 96, 128),
    ("B2_T16384_H16_D128", 2, 16384, 16, 128),
    ("B4_T2048_H16_D128", 4, 2048, 16, 128),
    ("B4_T4096_H64_D128", 4, 4096, 64, 128),
    ("B8_T2048_H32_D256", 8, 2048, 32, 256),
    ("B8_T1024_H8_D64", 8, 1024, 8, 64),
]

# bf16 matches FLA's prod baseline (flash-linear-attention/benchmarks/ops/registry.py:202).
DTYPE = torch.bfloat16


def _make_inputs(
    b: int, t: int, h: int, d: int, requires_grad: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(b, h, t, d, dtype=DTYPE, device=DEVICE, requires_grad=requires_grad)
    k = torch.randn(b, h, t, d, dtype=DTYPE, device=DEVICE, requires_grad=requires_grad)
    v = torch.randn(b, h, t, d, dtype=DTYPE, device=DEVICE, requires_grad=requires_grad)
    return q, k, v


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fwd-only", action="store_true", help="skip forward+backward bench"
    )
    args = parser.parse_args()

    rows: list[str] = []

    for name, b, t, h, d in SHAPES:
        scale = 1.0 / math.sqrt(d)

        q, k, v = _make_inputs(b, t, h, d, requires_grad=False)

        def helion_fwd(
            q: torch.Tensor = q,
            k: torch.Tensor = k,
            v: torch.Tensor = v,
            scale: float = scale,
        ) -> torch.Tensor:
            return chunk_linear_attn(q, k, v, scale=scale)

        def fla_fwd(
            q: torch.Tensor = q,
            k: torch.Tensor = k,
            v: torch.Tensor = v,
            scale: float = scale,
        ) -> torch.Tensor:
            return fla_chunk_linear_attn(q, k, v, scale=scale)

        h_fwd_ms = do_bench(helion_fwd)
        f_fwd_ms = do_bench(fla_fwd)
        fwd_pct = 100.0 * f_fwd_ms / h_fwd_ms

        row = f"{name:<22} {h_fwd_ms:>10.3f}ms {f_fwd_ms:>10.3f}ms {fwd_pct:>9.1f}%"

        if not args.fwd_only:
            q_g, k_g, v_g = _make_inputs(b, t, h, d, requires_grad=True)
            grad_out = torch.randn(b, h, t, d, dtype=DTYPE, device=DEVICE)

            def helion_fb(
                q: torch.Tensor = q_g,
                k: torch.Tensor = k_g,
                v: torch.Tensor = v_g,
                go: torch.Tensor = grad_out,
                scale: float = scale,
            ) -> None:
                o = chunk_linear_attn(q, k, v, scale=scale)
                o.backward(go)
                q.grad = k.grad = v.grad = None

            def fla_fb(
                q: torch.Tensor = q_g,
                k: torch.Tensor = k_g,
                v: torch.Tensor = v_g,
                go: torch.Tensor = grad_out,
                scale: float = scale,
            ) -> None:
                o = fla_chunk_linear_attn(q, k, v, scale=scale)
                o.backward(go)
                q.grad = k.grad = v.grad = None

            h_fb_ms = do_bench(helion_fb)
            f_fb_ms = do_bench(fla_fb)
            fb_pct = 100.0 * f_fb_ms / h_fb_ms
            row += f" {h_fb_ms:>10.3f}ms {f_fb_ms:>10.3f}ms {fb_pct:>9.1f}%"

        rows.append(row)

    header = f"{'Shape':<22} {'helion fwd':>11} {'fla fwd':>11} {'fwd %FLA':>10}"
    if not args.fwd_only:
        header += f" {'helion f+b':>11} {'fla f+b':>11} {'f+b %FLA':>10}"
    print()
    print(header)
    print("-" * (len(header) + 4))
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
