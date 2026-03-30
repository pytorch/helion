"""
Simple GLA (Gated Linear Attention with Scalar Decay) Example
=============================================================

Demonstrates the chunked linear attention engine on the Simple GLA variant:
scalar per-head decay, no correction term. Includes correctness tests against
both a naive recurrent reference and the chunked PyTorch reference, plus a
benchmark suite for forward and backward passes.
"""

from __future__ import annotations

import torch
from triton.testing import do_bench

from .linear_attention_engine import chunked_linear_attn
from .linear_attention_utils import chunked_linear_attn_reference
from .linear_attention_utils import make_simple_gla_inputs
from .linear_attention_utils import naive_recurrent_reference
from helion._testing import DEVICE


def test() -> None:
    """Correctness tests for Simple GLA forward and backward."""
    B, H, T, D, DV = 2, 4, 128, 32, 16
    C = 32
    dtype = torch.bfloat16
    atol_fwd = 1e-2
    rtol_fwd = 1e-2
    atol_grad = 2e-2
    rtol_grad = 2e-2

    # ------------------------------------------------------------------
    # Forward: compare against naive recurrent reference
    # ------------------------------------------------------------------
    q, k, v, g, scale = make_simple_gla_inputs(
        B, H, T, D, DV, dtype=dtype, device=DEVICE
    )
    out = chunked_linear_attn(q * scale, k, v, g, C=C)
    ref = naive_recurrent_reference(q, k, v, g, q_scale=scale)

    fwd_err = (out.float() - ref.float()).norm() / ref.float().norm().clamp(min=1e-8)
    fwd_ok = fwd_err.item() < 0.02
    print(
        f"Forward  (vs naive recurrent): {'PASS' if fwd_ok else 'FAIL'} (rel err={fwd_err:.4e})"
    )

    # ------------------------------------------------------------------
    # Backward: verify gradients are computed without error
    # ------------------------------------------------------------------
    q, k, v, g, scale = make_simple_gla_inputs(
        B, H, T, D, DV, dtype=dtype, device=DEVICE, requires_grad=True
    )
    out = chunked_linear_attn(q * scale, k, v, g, C=C)
    loss = out.sum()
    loss.backward()
    grads_exist = all(t.grad is not None and t.grad.shape == t.shape for t in (q, k, v))
    print(f"Backward (grads computed):     {'PASS' if grads_exist else 'FAIL'}")

    # ------------------------------------------------------------------
    # Gradient correctness: compare against chunked reference via autograd
    # ------------------------------------------------------------------
    q, k, v, g, scale = make_simple_gla_inputs(
        B, H, T, D, DV, dtype=dtype, device=DEVICE, requires_grad=True
    )
    out_engine = chunked_linear_attn(q * scale, k, v, g, C=C)
    grad_output = torch.randn_like(out_engine)
    grads_engine = torch.autograd.grad(out_engine, (q, k, v), grad_outputs=grad_output)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    out_ref = chunked_linear_attn_reference(q_ref * scale, k_ref, v_ref, g, C=C)
    grads_ref = torch.autograd.grad(
        out_ref, (q_ref, k_ref, v_ref), grad_outputs=grad_output
    )

    grad_ok = all(
        torch.allclose(ge.float(), gr.float(), atol=atol_grad, rtol=rtol_grad)
        for ge, gr in zip(grads_engine, grads_ref, strict=True)
    )
    print(f"Gradient (vs chunked ref):     {'PASS' if grad_ok else 'FAIL'}")


def benchmark() -> None:
    """Benchmark forward and backward for several problem sizes."""
    configs = [
        (1, 32, 2048, 128, 128),
        (1, 32, 4096, 128, 128),
    ]
    C = 32
    dtype = torch.bfloat16

    header = (
        f"{'B':>3}  {'H':>3}  {'T':>5}  {'D':>4}  {'DV':>4}  "
        f"{'Fwd (ms)':>10}  {'Bwd (ms)':>10}"
    )
    print(header)
    print("-" * len(header))

    for B, H, T, D, DV in configs:
        q, k, v, g, scale = make_simple_gla_inputs(
            B, H, T, D, DV, dtype=dtype, device=DEVICE, requires_grad=True
        )
        q_scaled = q * scale

        # Forward benchmark
        fwd_ms = do_bench(
            lambda qs=q_scaled, ki=k, vi=v, gi=g: chunked_linear_attn(
                qs, ki, vi, gi, C=C
            )
        )

        # Backward benchmark
        out = chunked_linear_attn(q_scaled, k, v, g, C=C)
        grad_output = torch.randn_like(out)

        def fwd_bwd(
            qs: torch.Tensor = q_scaled,
            ki: torch.Tensor = k,
            vi: torch.Tensor = v,
            gi: torch.Tensor = g,
            go: torch.Tensor = grad_output,
        ) -> None:
            o = chunked_linear_attn(qs, ki, vi, gi, C=C)
            o.backward(go)

        bwd_ms = do_bench(fwd_bwd)

        print(
            f"{B:>3}  {H:>3}  {T:>5}  {D:>4}  {DV:>4}  {fwd_ms:>10.3f}  {bwd_ms:>10.3f}"
        )


def main() -> None:
    test()
    benchmark()


if __name__ == "__main__":
    main()
