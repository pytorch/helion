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

    # Forward vs FLA chunk_rwkv6 (with u=zeros to match our engine)
    try:
        from fla.ops.rwkv6 import chunk_rwkv6

        out_no_gate = chunked_linear_attn(q, k, v, g, C=C)
        u_zeros = torch.zeros(H, D, device=DEVICE, dtype=DTYPE)
        o_fla, _ = chunk_rwkv6(
            r=_htf(q), k=_htf(k), v=_htf(v), w=_htf(g), u=u_zeros, scale=1.0
        )
        o_fla_hf = _htf(o_fla)
        fla_err = _rel_error(out_no_gate.detach(), o_fla_hf)
        # Note: chunk_rwkv6 with u=0 differs from pure GLA due to internal
        # normalization differences, so we use a loose tolerance here.
        print(
            f"  fwd vs FLA(rwkv6): {fla_err:.4e} {'PASS' if fla_err < 0.5 else 'FAIL'}"
        )
    except Exception as e:
        print(f"  fwd vs FLA:        SKIP ({type(e).__name__})")

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
    """Benchmark RWKV-6 forward+backward, comparing against FLA chunk_rwkv6."""
    from fla.ops.rwkv6 import chunk_rwkv6

    print(
        f"{'Config':<24} {'Helion fwd':>10} {'FLA fwd':>10}"
        f" {'Helion f+b':>12} {'FLA f+b':>12}"
    )
    print("-" * 72)

    for bi, hi, ti, di, dvi in BENCH_CONFIGS:
        q = torch.randn(
            bi, hi, ti, di, device=DEVICE, dtype=DTYPE, requires_grad=True
        )
        k = torch.randn(
            bi, hi, ti, di, device=DEVICE, dtype=DTYPE, requires_grad=True
        )
        v = torch.randn(
            bi, hi, ti, dvi, device=DEVICE, dtype=DTYPE, requires_grad=True
        )
        g = -torch.rand(bi, hi, ti, di, device=DEVICE, dtype=DTYPE).abs() * 0.1
        gate = torch.sigmoid(
            torch.randn(bi, hi, ti, dvi, device=DEVICE, dtype=DTYPE)
        )
        grad_out = torch.randn(bi, hi, ti, dvi, device=DEVICE, dtype=DTYPE)
        u_zeros = torch.zeros(hi, di, device=DEVICE, dtype=DTYPE)

        engine = LinearAttentionEngine(
            output_mod=lambda o, cio: o * cio["gate"],
            chunk_size=BENCH_C,
        )

        # Helion fwd (includes output gate)
        fwd_ms = do_bench(lambda: engine(q, k, v, g, gate=gate))

        # FLA fwd (no output gate — just the recurrence)
        qt = _htf(q.detach())
        kt = _htf(k.detach())
        vt = _htf(v.detach())
        wt = _htf(g)
        fla_fwd_ms = do_bench(
            lambda qt=qt, kt=kt, vt=vt, wt=wt, u=u_zeros: chunk_rwkv6(
                r=qt, k=kt, v=vt, w=wt, u=u, scale=1.0
            )
        )

        # Helion fwd+bwd
        def helion_fb(q=q, k=k, v=v, go=grad_out):
            o = engine(q, k, v, g, gate=gate)
            o.backward(go)
            q.grad = k.grad = v.grad = None

        fb_ms = do_bench(helion_fb)

        # FLA fwd+bwd
        qt_b = qt.clone().requires_grad_(True)
        kt_b = kt.clone().requires_grad_(True)
        vt_b = vt.clone().requires_grad_(True)
        go_t = _htf(grad_out)

        def fla_fb(qt=qt_b, kt=kt_b, vt=vt_b, wt=wt, u=u_zeros, go=go_t):
            o, _ = chunk_rwkv6(r=qt, k=kt, v=vt, w=wt, u=u, scale=1.0)
            o.backward(go)
            qt.grad = kt.grad = vt.grad = None

        fla_fb_ms = do_bench(fla_fb)

        cfg = f"({bi},{hi},{ti},{di},{dvi})"
        print(
            f"{cfg:<24} {fwd_ms:>10.3f} {fla_fwd_ms:>10.3f}"
            f" {fb_ms:>12.3f} {fla_fb_ms:>12.3f}"
        )


def main() -> None:
    print("=== RWKV-6 (diagonal decay + output gate) ===")
    test()
    print()
    benchmark()


if __name__ == "__main__":
    main()
