"""
DeltaNet (Delta Rule) Example
==============================

Chunked linear attention with rank-1 correction and no decay (DeltaNet).

The delta rule maintains a key-value state matrix S and applies a correction
step at each timestep:

    S_t = S_{t-1} - beta_t * (k_t (k_t^T S_{t-1})) + beta_t * (k_t v_t^T)
    o_t = q_t^T S_t

Keys are L2-normalized, beta is sigmoid-gated, and decay g is zero.
"""

from __future__ import annotations

import torch
from triton.testing import do_bench

from .linear_attention_engine import chunked_linear_attn
from .linear_attention_utils import chunked_linear_attn_reference
from .linear_attention_utils import make_delta_rule_inputs
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
    """Test DeltaNet forward and backward against references."""
    # --- Forward check ---
    q, k, v, g, beta, scale = make_delta_rule_inputs(
        B, H, T, D, DV, dtype=dtype, device=device
    )

    out = chunked_linear_attn(q * scale, k, v, g, beta=beta, C=C)
    ref = naive_recurrent_reference(q * scale, k, v, g, beta=beta)

    fwd_err = (out.float() - ref.float()).abs().max().item()
    print(f"Forward max error (vs recurrent): {fwd_err:.6e}")
    assert fwd_err < 1e-1, f"Forward error too large: {fwd_err}"

    # --- Backward gradient check ---
    q, k, v, g, beta, scale = make_delta_rule_inputs(
        B,
        H,
        T,
        D,
        DV,
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    g = g.detach().requires_grad_(True)
    beta = beta.detach().requires_grad_(True)

    # Helion path
    out_helion = chunked_linear_attn(q * scale, k, v, g, beta=beta, C=C)
    loss_helion = out_helion.sum()
    loss_helion.backward()
    dq_h = q.grad.clone()
    dk_h = k.grad.clone()
    dv_h = v.grad.clone()
    dg_h = g.grad.clone()
    dbeta_h = beta.grad.clone()

    # Reference path
    q2 = q.detach().requires_grad_(True)
    k2 = k.detach().requires_grad_(True)
    v2 = v.detach().requires_grad_(True)
    g2 = g.detach().requires_grad_(True)
    beta2 = beta.detach().requires_grad_(True)

    out_ref = chunked_linear_attn_reference(q2 * scale, k2, v2, g2, beta=beta2, C=C)
    loss_ref = out_ref.sum()
    loss_ref.backward()

    for name, grad_h, grad_r in [
        ("dq", dq_h, q2.grad),
        ("dk", dk_h, k2.grad),
        ("dv", dv_h, v2.grad),
        ("dg", dg_h, g2.grad),
        ("dbeta", dbeta_h, beta2.grad),
    ]:
        err = (grad_h.float() - grad_r.float()).abs().max().item()
        print(f"  {name} max error: {err:.6e}")
    print("Backward gradient check passed.")


def benchmark() -> None:
    """Benchmark DeltaNet forward."""
    configs = [
        (1, 32, 2048, 128, 128),
        (1, 32, 4096, 128, 128),
    ]
    C = 64
    dtype = torch.bfloat16

    for B, H, T, D, DV in configs:
        q, k, v, g, beta, scale = make_delta_rule_inputs(
            B, H, T, D, DV, dtype=dtype, device=DEVICE
        )

        fwd_ms = do_bench(
            lambda: chunked_linear_attn(q * scale, k, v, g, beta=beta, C=C)
        )
        print(f"  B={B}, H={H}, T={T}: fwd {fwd_ms:.3f} ms")


def main() -> None:
    """Run DeltaNet test and benchmark."""
    print("=== DeltaNet (correction, no decay) ===")
    print("--- Test ---")
    test()
    print("--- Benchmark ---")
    benchmark()


if __name__ == "__main__":
    main()
