"""
KDA (Kimi Delta Attention) Example
===================================

Diagonal (per-dimension) decay with rank-1 correction. This is the most
general variant — combines diagonal gating with the delta rule.

Note: On sm_100 (H200/B200), the Helion correction kernels with diagonal
decay hit Triton MLIR limitations. The engine automatically falls back to
the pure PyTorch reference for this combination.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from triton.testing import do_bench

from helion._testing import DEVICE

from .linear_attention_engine import LinearAttentionEngine
from .linear_attention_engine import chunked_linear_attn
from .linear_attention_utils import (
    chunked_linear_attn_reference,
    naive_recurrent_reference,
)

B, H, T, D, DV = 2, 4, 128, 32, 32
C = 32
DTYPE = torch.bfloat16
BENCH_CONFIGS = [(1, 32, 2048, 128, 128), (1, 32, 4096, 128, 128)]
BENCH_C = 64


def _rel_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).norm().item() / b.float().norm().clamp(
        min=1e-8
    ).item()


def _htf(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(1, 2).contiguous()


def test() -> None:
    """Forward + backward correctness for KDA (diagonal decay + correction)."""
    torch.manual_seed(42)
    scale = 1.0 / math.sqrt(D)

    q = torch.randn(B, H, T, D, device=DEVICE, dtype=DTYPE)
    k = F.normalize(
        torch.randn(B, H, T, D, device=DEVICE, dtype=DTYPE), dim=-1
    )
    v = torch.randn(B, H, T, DV, device=DEVICE, dtype=DTYPE)
    g = -torch.rand(B, H, T, D, device=DEVICE, dtype=DTYPE).abs() * 0.1
    beta = torch.sigmoid(
        torch.randn(B, H, T, device=DEVICE, dtype=DTYPE)
    )

    # Forward via engine (uses LinearAttentionEngine interface)
    engine = LinearAttentionEngine(
        q_mod=lambda q, cio: q * scale,
        beta_mod=lambda b, cio: b,
        chunk_size=C,
    )
    out = engine(q, k, v, g, beta=beta)

    # Forward vs recurrent reference
    ref = naive_recurrent_reference(
        q * scale, k, v, g, beta=beta
    )
    fwd_err = _rel_error(out.detach(), ref)
    assert fwd_err < 0.02, f"Forward error: {fwd_err}"
    print(f"  fwd vs recurrent: {fwd_err:.4e} PASS")

    # Forward vs FLA KDA
    try:
        from fla.ops.kda import chunk_kda

        o_fla, _ = chunk_kda(
            _htf(q),
            _htf(k),
            _htf(v),
            _htf(g),
            _htf(beta),
            scale=scale,
        )
        o_fla_hf = _htf(o_fla)
        fla_err = _rel_error(out.detach(), o_fla_hf)
        print(
            f"  fwd vs FLA:       {fla_err:.4e} {'PASS' if fla_err < 0.02 else 'FAIL'}"
        )
    except Exception as e:
        print(f"  fwd vs FLA:       SKIP ({type(e).__name__})")

    # Backward: verify gradients exist (uses autograd through reference)
    q1 = q.clone().requires_grad_(True)
    v1 = v.clone().requires_grad_(True)
    o1 = chunked_linear_attn(q1 * scale, k, v1, g, beta=beta, C=C)
    o1.sum().backward()
    assert q1.grad is not None, "q.grad is None"
    assert v1.grad is not None, "v.grad is None"
    print(f"  bwd grads exist:  PASS")

    # Backward vs chunked reference
    grad_out = torch.randn(B, H, T, DV, device=DEVICE, dtype=DTYPE)

    q2 = q.clone().requires_grad_(True)
    v2 = v.clone().requires_grad_(True)
    o2 = chunked_linear_attn(q2 * scale, k, v2, g, beta=beta, C=C)
    o2.backward(grad_out)

    q3 = q.clone().requires_grad_(True)
    v3 = v.clone().requires_grad_(True)
    o3 = chunked_linear_attn_reference(
        q3 * scale, k, v3, g, beta=beta, C=C
    )
    o3.backward(grad_out)

    for name, g1, g2 in [("dq", q2.grad, q3.grad), ("dv", v2.grad, v3.grad)]:
        err = _rel_error(g1, g2)
        assert err < 0.05, f"Backward {name} error: {err}"
        print(f"  bwd {name} vs ref: {err:.4e} PASS")

    print("All tests passed.")


def benchmark() -> None:
    """Benchmark KDA forward (uses reference fallback on sm_100)."""
    print(
        f"{'Config':<24} {'Fwd (ms)':>10}"
    )
    print("-" * 36)
    scale = 1.0 / math.sqrt(128)

    for bi, hi, ti, di, dvi in BENCH_CONFIGS:
        q = torch.randn(bi, hi, ti, di, device=DEVICE, dtype=DTYPE)
        k = F.normalize(
            torch.randn(bi, hi, ti, di, device=DEVICE, dtype=DTYPE), dim=-1
        )
        v = torch.randn(bi, hi, ti, dvi, device=DEVICE, dtype=DTYPE)
        g = -torch.rand(bi, hi, ti, di, device=DEVICE, dtype=DTYPE).abs() * 0.1
        beta = torch.sigmoid(
            torch.randn(bi, hi, ti, device=DEVICE, dtype=DTYPE)
        )

        fwd_ms = do_bench(
            lambda q=q, k=k, v=v, g=g, beta=beta: chunked_linear_attn(
                q * scale, k, v, g, beta=beta, C=BENCH_C
            )
        )
        label = f"B={bi},H={hi},T={ti}"
        print(f"  {label:<22} {fwd_ms:>8.3f}ms")


def main() -> None:
    print("=== KDA (diagonal decay + correction) ===")
    test()
    print()
    benchmark()


if __name__ == "__main__":
    main()
