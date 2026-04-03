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
import warnings

import torch
import torch.nn.functional as F
from triton.testing import do_bench

from .linear_attention_engine import LinearAttentionEngine
from .linear_attention_engine import chunked_linear_attn
from .linear_attention_engine import recurrent_step
from .linear_attention_utils import chunked_linear_attn_reference
from .linear_attention_utils import head_to_time_first as _htf
from .linear_attention_utils import naive_recurrent_reference
from .linear_attention_utils import rel_error as _rel_error
from helion._testing import DEVICE

B, H, T, D, DV = 2, 4, 128, 32, 32
C = 32
DTYPE = torch.bfloat16
BENCH_CONFIGS = [(1, 32, 2048, 128, 128), (1, 32, 4096, 128, 128)]
BENCH_C = 64


def test() -> None:
    """Forward + backward correctness for KDA (diagonal decay + correction)."""
    torch.manual_seed(42)
    scale = 1.0 / math.sqrt(D)

    q = torch.randn(B, H, T, D, device=DEVICE, dtype=DTYPE)
    k = F.normalize(torch.randn(B, H, T, D, device=DEVICE, dtype=DTYPE), dim=-1)
    v = torch.randn(B, H, T, DV, device=DEVICE, dtype=DTYPE)
    g = -torch.rand(B, H, T, D, device=DEVICE, dtype=DTYPE).abs() * 0.1
    beta = torch.sigmoid(torch.randn(B, H, T, device=DEVICE, dtype=DTYPE))

    # Forward via engine (uses LinearAttentionEngine interface)
    engine = LinearAttentionEngine(
        q_mod=lambda q, cio: q * scale,
        beta_mod=lambda b, cio: b,
        chunk_size=C,
    )
    out = engine(q, k, v, g, beta=beta)

    # Forward vs recurrent reference
    ref = naive_recurrent_reference(q * scale, k, v, g, beta=beta)
    fwd_err = _rel_error(out.detach(), ref)
    assert fwd_err < 0.02, f"Forward error: {fwd_err}"
    print(f"  fwd vs recurrent: {fwd_err:.4e} PASS")

    # Forward vs FLA KDA
    try:
        from fla.ops.kda import chunk_kda  # pyrefly: ignore

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
    print("  bwd grads exist:  PASS")

    # Backward vs chunked reference
    grad_out = torch.randn(B, H, T, DV, device=DEVICE, dtype=DTYPE)

    q2 = q.clone().requires_grad_(True)
    v2 = v.clone().requires_grad_(True)
    o2 = chunked_linear_attn(q2 * scale, k, v2, g, beta=beta, C=C)
    o2.backward(grad_out)

    q3 = q.clone().requires_grad_(True)
    v3 = v.clone().requires_grad_(True)
    o3 = chunked_linear_attn_reference(q3 * scale, k, v3, g, beta=beta, C=C)
    o3.backward(grad_out)

    for name, g1, g2 in [("dq", q2.grad, q3.grad), ("dv", v2.grad, v3.grad)]:
        err = _rel_error(g1, g2)
        assert err < 0.05, f"Backward {name} error: {err}"
        print(f"  bwd {name} vs ref: {err:.4e} PASS")

    # === Recurrent step: compare step-by-step vs chunked ===
    torch.manual_seed(42)
    q_rec = torch.randn(B, H, T, D, device=DEVICE, dtype=DTYPE)
    k_rec = F.normalize(torch.randn(B, H, T, D, device=DEVICE, dtype=DTYPE), dim=-1)
    v_rec = torch.randn(B, H, T, DV, device=DEVICE, dtype=DTYPE)
    g_rec = -torch.rand(B, H, T, D, device=DEVICE, dtype=DTYPE).abs() * 0.1
    beta_rec = torch.sigmoid(torch.randn(B, H, T, device=DEVICE, dtype=DTYPE))
    scale_rec = 1.0 / math.sqrt(D)

    o_chunked = chunked_linear_attn(
        q_rec * scale_rec, k_rec, v_rec, g_rec, beta=beta_rec, C=C
    )

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
            beta_val=beta_rec[:, :, t : t + 1],
        )
        o_steps.append(o_t)
    o_recurrent = torch.cat(o_steps, dim=2)

    rec_err = _rel_error(o_chunked, o_recurrent)
    assert rec_err < 0.05, f"Recurrent vs chunked error: {rec_err}"
    print(f"  recurrent step:   {rec_err:.4e} PASS")

    print("All tests passed.")


def benchmark() -> None:
    """Benchmark KDA forward+backward, comparing against FLA."""
    try:
        from fla.ops.kda import chunk_kda  # pyrefly: ignore
    except ImportError:
        warnings.warn("fla not installed, skipping benchmark", stacklevel=1)
        return

    scale = 1.0 / math.sqrt(128)

    print(
        f"{'Config':<24} {'Helion fwd':>10} {'FLA fwd':>10}"
        f" {'Helion f+b':>12} {'FLA f+b':>12}"
    )
    print("-" * 72)

    for bi, hi, ti, di, dvi in BENCH_CONFIGS:
        q = torch.randn(bi, hi, ti, di, device=DEVICE, dtype=DTYPE, requires_grad=True)
        k = (
            F.normalize(torch.randn(bi, hi, ti, di, device=DEVICE, dtype=DTYPE), dim=-1)
            .detach()
            .requires_grad_(True)
        )
        v = torch.randn(bi, hi, ti, dvi, device=DEVICE, dtype=DTYPE, requires_grad=True)
        g = -torch.rand(bi, hi, ti, di, device=DEVICE, dtype=DTYPE).abs() * 0.1
        beta = torch.sigmoid(torch.randn(bi, hi, ti, device=DEVICE, dtype=DTYPE))
        grad_out = torch.randn(bi, hi, ti, dvi, device=DEVICE, dtype=DTYPE)

        qt = _htf(q.detach())
        kt = _htf(k.detach())
        vt = _htf(v.detach())
        gt = _htf(g)
        bt = _htf(beta)

        fwd_ms = do_bench(
            lambda q=q, k=k, v=v, g=g, beta=beta: chunked_linear_attn(
                q * scale, k, v, g, beta=beta, C=BENCH_C
            )
        )
        fla_fwd_ms = do_bench(
            lambda qt=qt, kt=kt, vt=vt, gt=gt, bt=bt: chunk_kda(
                qt, kt, vt, gt, bt, scale=scale
            )
        )

        def helion_fb(
            q: torch.Tensor = q,
            k: torch.Tensor = k,
            v: torch.Tensor = v,
            g: torch.Tensor = g,
            beta: torch.Tensor = beta,
            go: torch.Tensor = grad_out,
            sc: float = scale,
        ) -> None:
            o = chunked_linear_attn(q * sc, k, v, g, beta=beta, C=BENCH_C)
            o.backward(go)
            q.grad = k.grad = v.grad = None

        fb_ms = do_bench(helion_fb)

        # NOTE: FLA's KDA backward crashes Triton's autotuner on H200 (CUDA
        # IMA), which would poison the CUDA context for all subsequent work.
        # We skip the FLA backward benchmark to keep the process healthy.
        cfg = f"({bi},{hi},{ti},{di},{dvi})"
        print(
            f"{cfg:<24} {fwd_ms:>10.3f} {fla_fwd_ms:>10.3f}"
            f" {fb_ms:>12.3f} {'(skip)':>12}"
        )


def main() -> None:
    print("=== KDA (diagonal decay + correction) ===")
    test()
    print()
    benchmark()


if __name__ == "__main__":
    main()
