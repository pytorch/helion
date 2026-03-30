"""
Gated Delta Rule Example
========================

Demonstrates Gated DeltaNet (correction + scalar decay) using the
chunked linear attention engine.

Gated DeltaNet combines:
  - L2-normalized keys
  - Scalar gated decay: g = logsigmoid(noise)
  - Correction via beta = sigmoid(noise)
"""

from __future__ import annotations

import torch
from triton.testing import do_bench

from .linear_attention_engine import chunked_linear_attn
from .linear_attention_utils import chunked_linear_attn_reference
from .linear_attention_utils import make_gated_delta_rule_inputs
from .linear_attention_utils import naive_recurrent_reference
from helion._testing import DEVICE


def test(
    B: int = 2,
    H: int = 4,
    T: int = 128,
    D: int = 32,
    DV: int = 32,
    C: int = 32,
    dtype: torch.dtype = torch.bfloat16,
    device: str = DEVICE,
) -> None:
    """Test Gated DeltaNet forward and backward correctness."""
    torch.manual_seed(0)

    # ── Forward correctness ──────────────────────────────────────────────
    q, k, v, g, beta, scale = make_gated_delta_rule_inputs(
        B, H, T, D, DV, dtype=dtype, device=device
    )

    out = chunked_linear_attn(q * scale, k, v, g, beta=beta, C=C)
    ref = naive_recurrent_reference(q, k, v, g, beta=beta, q_scale=scale)

    fwd_err = (out.float() - ref.float()).abs().max().item()
    print(f"Forward max error (vs recurrent): {fwd_err:.6e}")
    assert fwd_err < 1e-1, f"Forward error too large: {fwd_err}"

    # ── Backward correctness ─────────────────────────────────────────────
    q, k, v, g, beta, scale = make_gated_delta_rule_inputs(
        B, H, T, D, DV, dtype=dtype, device=device, requires_grad=True
    )

    out = chunked_linear_attn(q * scale, k, v, g, beta=beta, C=C)
    loss = out.sum()
    loss.backward()

    grad_q = q.grad.clone()
    grad_k = k.grad.clone()
    grad_v = v.grad.clone()

    q.grad, k.grad, v.grad = None, None, None

    out_ref = chunked_linear_attn_reference(q * scale, k, v, g, beta=beta, C=C)
    loss_ref = out_ref.sum()
    loss_ref.backward()

    for name, g_actual, g_ref in [
        ("dq", grad_q, q.grad),
        ("dk", grad_k, k.grad),
        ("dv", grad_v, v.grad),
    ]:
        err = (g_actual.float() - g_ref.float()).abs().max().item()
        print(f"Backward {name} max error (vs reference): {err:.6e}")
        assert err < 1e-1, f"Backward {name} error too large: {err}"

    print("All checks passed.")


def benchmark() -> None:
    """Benchmark Gated DeltaNet forward and backward."""
    configs = [
        (1, 32, 2048, 128, 128),
        (1, 32, 4096, 128, 128),
    ]
    C = 64
    dtype = torch.bfloat16

    print("\n=== Gated DeltaNet Benchmark ===")
    for B, H, T, D, DV in configs:
        q, k, v, g, beta, scale = make_gated_delta_rule_inputs(
            B, H, T, D, DV, dtype=dtype, device=DEVICE
        )

        fwd_ms = do_bench(
            lambda: chunked_linear_attn(q * scale, k, v, g, beta=beta, C=C)
        )
        print(f"  B={B}, H={H}, T={T}: fwd {fwd_ms:.3f} ms")


def main() -> None:
    """Run tests and benchmarks for Gated DeltaNet."""
    print("=== Gated DeltaNet Test ===")
    test()
    benchmark()


if __name__ == "__main__":
    main()
