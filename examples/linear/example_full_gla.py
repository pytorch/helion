"""
Full GLA (Diagonal Decay) Example
==================================

Demonstrates chunked linear attention with per-dimension (diagonal) decay gates.
Includes correctness tests against a naive recurrent reference, FLA, and chunked
reference backward, plus a benchmark suite comparing against FLA.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from triton.testing import do_bench

from .linear_attention_engine import chunked_linear_attn
from .linear_attention_engine import recurrent_step
from .linear_attention_utils import chunked_linear_attn_reference
from .linear_attention_utils import make_full_gla_inputs
from .linear_attention_utils import naive_recurrent_reference
from helion._testing import DEVICE

# Test/benchmark config
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
    """Head-first [B,H,T,...] -> time-first [B,T,H,...] for FLA."""
    return x.transpose(1, 2).contiguous()


def test() -> None:
    """Forward + backward correctness vs reference and FLA."""
    torch.manual_seed(42)

    # === Make inputs ===
    q, k, v, g, scale = make_full_gla_inputs(B, H, T, D, DV, dtype=DTYPE, device=DEVICE)

    # === Forward: vs naive recurrent reference ===
    out = chunked_linear_attn(q * scale, k, v, g, C=C)
    ref = naive_recurrent_reference(q * scale, k, v, g)
    fwd_err = _rel_error(out, ref)
    assert fwd_err < 0.02, f"Forward error: {fwd_err}"
    print(f"  fwd vs recurrent: {fwd_err:.4e} PASS")

    # === Forward: vs FLA ===
    from fla.ops.gla import chunk_gla

    o_fla, _ = chunk_gla(_htf(q), _htf(k), _htf(v), _htf(g), scale=scale)
    o_fla_hf = o_fla.transpose(1, 2).contiguous()
    fla_err = _rel_error(out, o_fla_hf)
    print(f"  fwd vs FLA:       {fla_err:.4e} {'PASS' if fla_err < 0.02 else 'FAIL'}")

    # === Backward: vs chunked reference ===
    grad_out = torch.randn(B, H, T, DV, device=DEVICE, dtype=DTYPE)
    q1 = q.clone().requires_grad_(True)
    k1 = k.clone().requires_grad_(True)
    v1 = v.clone().requires_grad_(True)
    o1 = chunked_linear_attn(q1 * scale, k1, v1, g, C=C)
    o1.backward(grad_out)

    q2 = q.clone().requires_grad_(True)
    k2 = k.clone().requires_grad_(True)
    v2 = v.clone().requires_grad_(True)
    o2 = chunked_linear_attn_reference(q2 * scale, k2, v2, g, C=C)
    o2.backward(grad_out)

    for name, g1, g2 in [
        ("dq", q1.grad, q2.grad),
        ("dk", k1.grad, k2.grad),
        ("dv", v1.grad, v2.grad),
    ]:
        err = _rel_error(g1, g2)
        assert err < 0.05, f"Backward {name} error: {err}"
        print(f"  bwd {name} vs ref: {err:.4e} PASS")

    # === Backward: vs FLA (dq comparison) ===
    q3 = q.clone().requires_grad_(True)
    k3 = k.clone().requires_grad_(True)
    v3 = v.clone().requires_grad_(True)
    o3 = chunked_linear_attn(q3 * scale, k3, v3, g, C=C)
    o3.backward(grad_out)

    q4 = _htf(q).clone().requires_grad_(True)
    k4 = _htf(k).clone().requires_grad_(True)
    v4 = _htf(v).clone().requires_grad_(True)
    o4, _ = chunk_gla(q4, k4, v4, _htf(g), scale=scale)
    o4.backward(_htf(grad_out))

    dq_err = _rel_error(q3.grad, q4.grad.transpose(1, 2).contiguous())
    print(f"  bwd dq vs FLA:    {dq_err:.4e} {'PASS' if dq_err < 0.05 else 'FAIL'}")
    dk_err = _rel_error(k3.grad, k4.grad.transpose(1, 2).contiguous())
    dv_err = _rel_error(v3.grad, v4.grad.transpose(1, 2).contiguous())
    print(f"  bwd dk vs FLA:    {dk_err:.4e} (info)")
    print(f"  bwd dv vs FLA:    {dv_err:.4e} (info)")

    # === Recurrent step: compare step-by-step vs chunked ===
    torch.manual_seed(42)
    q_rec = torch.randn(B, H, T, D, device=DEVICE, dtype=DTYPE)
    k_rec = torch.randn(B, H, T, D, device=DEVICE, dtype=DTYPE)
    v_rec = torch.randn(B, H, T, DV, device=DEVICE, dtype=DTYPE)
    g_rec = F.logsigmoid(torch.randn(B, H, T, D, device=DEVICE, dtype=DTYPE))
    scale_rec = 1.0 / math.sqrt(D)

    o_chunked = chunked_linear_attn(q_rec * scale_rec, k_rec, v_rec, g_rec, C=C)

    state = torch.zeros(B, H, D, DV, device=DEVICE, dtype=torch.float32)
    o_steps = []
    for t in range(T):
        alpha = torch.exp(g_rec[:, :, t : t + 1])  # [B,H,1,D]
        o_t, state = recurrent_step(
            q_rec[:, :, t : t + 1] * scale_rec,
            k_rec[:, :, t : t + 1],
            v_rec[:, :, t : t + 1],
            state,
            alpha=alpha,
        )
        o_steps.append(o_t)
    o_recurrent = torch.cat(o_steps, dim=2)

    rec_err = _rel_error(o_chunked, o_recurrent)
    assert rec_err < 0.05, f"Recurrent vs chunked error: {rec_err}"
    print(f"  recurrent step:   {rec_err:.4e} PASS")

    print("All tests passed.")


def benchmark() -> None:
    """Benchmark forward and fwd+bwd, comparing against FLA."""
    from fla.ops.gla import chunk_gla

    print(
        f"{'Config':<24} {'Helion fwd':>10} {'FLA fwd':>10}"
        f" {'Helion f+b':>12} {'FLA f+b':>12}"
    )
    print("-" * 72)

    for bi, hi, ti, di, dvi in BENCH_CONFIGS:
        q, k, v, g, scale = make_full_gla_inputs(
            bi, hi, ti, di, dvi, dtype=DTYPE, device=DEVICE, requires_grad=True
        )
        grad_out = torch.randn(bi, hi, ti, dvi, device=DEVICE, dtype=DTYPE)
        qt = _htf(q)
        kt = _htf(k)
        vt = _htf(v)
        gt = _htf(g)
        go_t = _htf(grad_out)

        def helion_fwd(
            q: torch.Tensor = q,
            ki: torch.Tensor = k,
            vi: torch.Tensor = v,
            gi: torch.Tensor = g,
            sc: float = scale,
        ) -> torch.Tensor:
            return chunked_linear_attn(q * sc, ki, vi, gi, C=BENCH_C)

        fwd_ms = do_bench(helion_fwd)

        def fla_fwd(
            qt: torch.Tensor = qt,
            kt: torch.Tensor = kt,
            vt: torch.Tensor = vt,
            gt: torch.Tensor = gt,
            sc: float = scale,
        ) -> torch.Tensor:
            o, _ = chunk_gla(qt, kt, vt, gt, scale=sc)
            return o

        fla_fwd_ms = do_bench(fla_fwd)

        def helion_fb(
            q: torch.Tensor = q,
            ki: torch.Tensor = k,
            vi: torch.Tensor = v,
            gi: torch.Tensor = g,
            go: torch.Tensor = grad_out,
            sc: float = scale,
        ) -> None:
            o = chunked_linear_attn(q * sc, ki, vi, gi, C=BENCH_C)
            o.backward(go)
            q.grad = ki.grad = vi.grad = None

        fb_ms = do_bench(helion_fb)

        def fla_fb(
            qt: torch.Tensor = qt,
            kt: torch.Tensor = kt,
            vt: torch.Tensor = vt,
            gt: torch.Tensor = gt,
            go: torch.Tensor = go_t,
            sc: float = scale,
        ) -> None:
            o, _ = chunk_gla(qt, kt, vt, gt, scale=sc)
            o.backward(go)
            qt.grad = kt.grad = vt.grad = None

        fla_fb_ms = do_bench(fla_fb)

        cfg = f"({bi},{hi},{ti},{di},{dvi})"
        print(
            f"{cfg:<24} {fwd_ms:>10.3f} {fla_fwd_ms:>10.3f}"
            f" {fb_ms:>12.3f} {fla_fb_ms:>12.3f}"
        )


def main() -> None:
    print("=== Full GLA ===")
    test()
    print()
    benchmark()


if __name__ == "__main__":
    main()
