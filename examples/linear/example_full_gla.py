"""
Full GLA Example (Diagonal Decay, No Correction)
=================================================

Demonstrates chunked linear attention with per-dimension (diagonal) decay gates,
validating forward correctness and backward gradients against reference implementations.
"""

from __future__ import annotations

import torch
from triton.testing import do_bench

from .linear_attention_engine import chunked_linear_attn
from .linear_attention_utils import chunked_linear_attn_reference
from .linear_attention_utils import make_full_gla_inputs
from .linear_attention_utils import naive_recurrent_reference
from helion._testing import DEVICE


def test() -> None:
    B, H, T, D, DV, C = 2, 4, 128, 32, 16, 32
    dtype = torch.bfloat16

    # --- Forward correctness vs naive recurrent reference ---
    q, k, v, g, scale = make_full_gla_inputs(B, H, T, D, DV, dtype=dtype, device=DEVICE)
    # g has shape [B, H, T, D] for full GLA (diagonal decay)
    assert g.shape == (B, H, T, D), f"Expected g shape {(B, H, T, D)}, got {g.shape}"

    out = chunked_linear_attn(q * scale, k, v, g, C=C)
    ref = naive_recurrent_reference(q * scale, k, v, g)

    fwd_err = (out.float() - ref.float()).abs().max().item()
    fwd_rel = (out.float() - ref.float()).norm() / ref.float().norm().clamp(min=1e-8)
    print(f"Forward max-abs error: {fwd_err:.6e}, rel error: {fwd_rel:.6e}")
    assert fwd_rel < 0.02, f"Forward rel error too large: {fwd_rel}"

    # --- Backward gradient check vs chunked reference ---
    q, k, v, g, scale = make_full_gla_inputs(
        B, H, T, D, DV, dtype=dtype, device=DEVICE, requires_grad=True
    )
    g = g.detach().requires_grad_(True)

    out = chunked_linear_attn(q * scale, k, v, g, C=C)
    loss = out.sum()
    loss.backward()

    dq_actual = q.grad.clone()
    dk_actual = k.grad.clone()
    dv_actual = v.grad.clone()
    dg_actual = g.grad.clone()

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    g_ref = g.detach().clone().requires_grad_(True)

    out_ref = chunked_linear_attn_reference(q_ref * scale, k_ref, v_ref, g_ref, C=C)
    loss_ref = out_ref.sum()
    loss_ref.backward()

    for name, actual, expected in [
        ("dq", dq_actual, q_ref.grad),
        ("dk", dk_actual, k_ref.grad),
        ("dv", dv_actual, v_ref.grad),
        ("dg", dg_actual, g_ref.grad),
    ]:
        err = (actual.float() - expected.float()).abs().max().item()
        print(f"Backward {name} max error vs chunked reference: {err:.6e}")
        assert err < 1e-1, f"Backward {name} error too large: {err}"

    print("All tests passed.")


def benchmark() -> None:
    configs = [
        (1, 32, 2048, 128, 128),
        (1, 32, 4096, 128, 128),
    ]
    C = 64
    dtype = torch.bfloat16

    for B, H, T, D, DV in configs:
        q, k, v, g, scale = make_full_gla_inputs(
            B, H, T, D, DV, dtype=dtype, device=DEVICE, requires_grad=True
        )
        g = g.detach().requires_grad_(True)

        def fwd():
            return chunked_linear_attn(q * scale, k, v, g, C=C)

        def fwd_bwd():
            out = chunked_linear_attn(q * scale, k, v, g, C=C)
            out.sum().backward(retain_graph=True)

        fwd_ms = do_bench(fwd)
        fwd_bwd_ms = do_bench(fwd_bwd)

        print(
            f"B={B}, H={H}, T={T}, D={D}, DV={DV}: "
            f"fwd {fwd_ms:.3f} ms, fwd+bwd {fwd_bwd_ms:.3f} ms"
        )


def main() -> None:
    test()
    benchmark()


if __name__ == "__main__":
    main()
