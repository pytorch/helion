from __future__ import annotations

import torch
from triton.testing import do_bench

from .linear_attention_engine import chunked_linear_attn
from .linear_attention_utils import chunked_linear_attn_reference
from .linear_attention_utils import make_mamba2_inputs
from .linear_attention_utils import naive_recurrent_reference
from helion._testing import DEVICE


def _rel_error(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = b.float().norm().clamp(min=1e-8).item()
    return (a.float() - b.float()).norm().item() / denom


def test():
    B, H, T, D, DV, C = 2, 4, 128, 32, 16, 32
    dtype = torch.bfloat16

    # --- Forward correctness ---
    q, k, v, g, scale = make_mamba2_inputs(B, H, T, D, DV, dtype=dtype, device=DEVICE)

    out = chunked_linear_attn(q, k, v, g, C=C)
    ref = chunked_linear_attn_reference(q, k, v, g, C=C)
    rec = naive_recurrent_reference(q, k, v, g, q_scale=scale)

    err_vs_ref = _rel_error(out, ref)
    err_vs_rec = _rel_error(out, rec)
    print(f"[mamba2] fwd  rel-err vs chunked ref: {err_vs_ref:.4e}")
    print(f"[mamba2] fwd  rel-err vs recurrent  : {err_vs_rec:.4e}")
    assert err_vs_ref < 0.02, f"Forward error too large: {err_vs_ref}"

    # --- Backward correctness ---
    q, k, v, g, scale = make_mamba2_inputs(
        B, H, T, D, DV, dtype=dtype, device=DEVICE, requires_grad=True
    )

    out = chunked_linear_attn(q, k, v, g, C=C)
    loss = out.sum()
    loss.backward()

    assert q.grad is not None, "q.grad is None"
    assert k.grad is not None, "k.grad is None"
    assert v.grad is not None, "v.grad is None"

    # Reference grads
    q2, k2, v2, g2, _ = make_mamba2_inputs(
        B, H, T, D, DV, dtype=dtype, device=DEVICE, requires_grad=True
    )
    q2.data.copy_(q.data)
    k2.data.copy_(k.data)
    v2.data.copy_(v.data)
    g2 = g.detach().clone()

    ref_out = chunked_linear_attn_reference(q2, k2, v2, g2, C=C)
    ref_out.sum().backward()

    for name, grad, ref_grad in [
        ("q", q.grad, q2.grad),
        ("k", k.grad, k2.grad),
        ("v", v.grad, v2.grad),
    ]:
        err = _rel_error(grad, ref_grad)
        print(f"[mamba2] bwd  grad-{name} rel-err: {err:.4e}")
        assert err < 0.05, f"Backward grad-{name} error too large: {err}"

    print("[mamba2] all tests passed")


def benchmark():
    B, H, T, D, DV, C = 2, 4, 1024, 64, 64, 64
    dtype = torch.bfloat16

    q, k, v, g, scale = make_mamba2_inputs(B, H, T, D, DV, dtype=dtype, device=DEVICE)

    fwd_ms = do_bench(lambda: chunked_linear_attn(q, k, v, g, C=C))
    print(f"[mamba2] fwd  {fwd_ms:.3f} ms  (B={B}, H={H}, T={T}, D={D}, DV={DV})")

    q, k, v, g, scale = make_mamba2_inputs(
        B, H, T, D, DV, dtype=dtype, device=DEVICE, requires_grad=True
    )
    chunked_linear_attn(q, k, v, g, C=C)  # warmup

    def fwd_bwd():
        out_ = chunked_linear_attn(q, k, v, g, C=C)
        out_.sum().backward(retain_graph=True)

    fwd_bwd_ms = do_bench(fwd_bwd)
    print(f"[mamba2] fwd+bwd  {fwd_bwd_ms:.3f} ms")


def main():
    test()
    benchmark()


if __name__ == "__main__":
    main()
