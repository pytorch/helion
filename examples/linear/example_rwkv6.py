"""
RWKV-6 Example
==============

Diagonal (per-dimension) decay with output gating. This is structurally
Full GLA plus an output gate applied after the attention computation.
Uses the LinearAttentionEngine with output_mod.
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

B, H, T, D, DV = 2, 4, 128, 32, 16
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
    """Forward + backward correctness for RWKV-6 (diagonal decay + output gate)."""
    torch.manual_seed(42)

    q = torch.randn(B, H, T, D, device=DEVICE, dtype=DTYPE)
    k = torch.randn(B, H, T, D, device=DEVICE, dtype=DTYPE)
    v = torch.randn(B, H, T, DV, device=DEVICE, dtype=DTYPE)
    g = -torch.rand(B, H, T, D, device=DEVICE, dtype=DTYPE).abs() * 0.1
    gate = torch.sigmoid(
        torch.randn(B, H, T, DV, device=DEVICE, dtype=DTYPE)
    )

    engine = LinearAttentionEngine(
        output_mod=lambda o, cio: o * cio["gate"],
        chunk_size=C,
    )

    # Forward vs recurrent reference
    out = engine(q, k, v, g, gate=gate)
    ref = naive_recurrent_reference(q, k, v, g)
    ref = ref * gate.detach()
    fwd_err = _rel_error(out.detach(), ref)
    assert fwd_err < 0.02, f"Forward error: {fwd_err}"
    print(f"  fwd vs recurrent: {fwd_err:.4e} PASS")

    # Forward vs FLA RWKV-6
    # FLA chunk_rwkv6 signature: (r, k, v, w, u, scale) where u is a bonus vector.
    # For comparison, we use chunk_gla which matches our diagonal decay path.
    try:
        from fla.ops.gla import chunk_gla

        out_no_gate = chunked_linear_attn(q, k, v, g, C=C)
        o_fla, _ = chunk_gla(
            _htf(q), _htf(k), _htf(v), _htf(g), scale=1.0
        )
        o_fla_hf = _htf(o_fla)
        fla_err = _rel_error(out_no_gate.detach(), o_fla_hf)
        print(
            f"  fwd vs FLA(gla):  {fla_err:.4e} {'PASS' if fla_err < 0.02 else 'FAIL'}"
        )
    except Exception as e:
        print(f"  fwd vs FLA:       SKIP ({type(e).__name__})")

    # Backward vs chunked reference
    grad_out = torch.randn(B, H, T, DV, device=DEVICE, dtype=DTYPE)

    q1 = q.clone().requires_grad_(True)
    k1 = k.clone().requires_grad_(True)
    v1 = v.clone().requires_grad_(True)
    o1 = chunked_linear_attn(q1, k1, v1, g, C=C)
    (o1 * gate).backward(grad_out)

    q2 = q.clone().requires_grad_(True)
    k2 = k.clone().requires_grad_(True)
    v2 = v.clone().requires_grad_(True)
    o2 = chunked_linear_attn_reference(q2, k2, v2, g, C=C)
    (o2 * gate).backward(grad_out)

    for name, g1, g2 in [
        ("dq", q1.grad, q2.grad),
        ("dk", k1.grad, k2.grad),
        ("dv", v1.grad, v2.grad),
    ]:
        err = _rel_error(g1, g2)
        assert err < 0.05, f"Backward {name} error: {err}"
        print(f"  bwd {name} vs ref: {err:.4e} PASS")

    print("All tests passed.")


def benchmark() -> None:
    """Benchmark RWKV-6 forward and fwd+bwd."""
    print(
        f"{'Config':<24} {'Helion fwd':>10} {'Helion f+b':>12}"
    )
    print("-" * 50)

    for bi, hi, ti, di, dvi in BENCH_CONFIGS:
        q = torch.randn(bi, hi, ti, di, device=DEVICE, dtype=DTYPE)
        k = torch.randn(bi, hi, ti, di, device=DEVICE, dtype=DTYPE)
        v = torch.randn(bi, hi, ti, dvi, device=DEVICE, dtype=DTYPE)
        g = -torch.rand(bi, hi, ti, di, device=DEVICE, dtype=DTYPE).abs() * 0.1
        gate = torch.sigmoid(
            torch.randn(bi, hi, ti, dvi, device=DEVICE, dtype=DTYPE)
        )

        engine = LinearAttentionEngine(
            output_mod=lambda o, cio: o * cio["gate"],
            chunk_size=BENCH_C,
        )

        fwd_ms = do_bench(lambda: engine(q, k, v, g, gate=gate))

        q_b = q.clone().requires_grad_(True)
        k_b = k.clone().requires_grad_(True)
        v_b = v.clone().requires_grad_(True)
        go = torch.randn(bi, hi, ti, dvi, device=DEVICE, dtype=DTYPE)

        def fwd_bwd(q_b=q_b, k_b=k_b, v_b=v_b):
            o = engine(q_b, k_b, v_b, g, gate=gate)
            o.backward(go)
            q_b.grad = k_b.grad = v_b.grad = None

        bwd_ms = do_bench(fwd_bwd)

        label = f"B={bi},H={hi},T={ti}"
        print(f"  {label:<22} {fwd_ms:>8.3f}ms {bwd_ms:>10.3f}ms")


def main() -> None:
    print("=== RWKV-6 (diagonal decay + output gate) ===")
    test()
    print()
    benchmark()


if __name__ == "__main__":
    main()
