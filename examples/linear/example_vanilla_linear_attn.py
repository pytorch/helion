from __future__ import annotations

import torch
from triton.testing import do_bench

from .linear_attention_engine import chunked_linear_attn
from .linear_attention_utils import chunked_linear_attn_reference
from .linear_attention_utils import make_vanilla_linear_attn_inputs
from .linear_attention_utils import naive_recurrent_reference
from helion._testing import DEVICE


def _rel_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).norm().item() / b.float().norm().clamp(
        min=1e-8
    ).item()


def test():
    B, H, T, D, DV, C = 2, 4, 128, 32, 16, 32
    dtype = torch.bfloat16

    # --- Forward correctness ---
    q, k, v, g, scale = make_vanilla_linear_attn_inputs(
        B, H, T, D, DV, dtype=dtype, device=DEVICE
    )
    q_s = q * scale

    out = chunked_linear_attn(q_s, k, v, g, C=C)
    ref = chunked_linear_attn_reference(q_s, k, v, g, C=C)
    rec = naive_recurrent_reference(q, k, v, g, q_scale=scale)

    err_vs_ref = _rel_error(out, ref)
    err_vs_rec = _rel_error(out, rec)
    print(f"[vanilla] fwd  rel-err vs chunked ref: {err_vs_ref:.4e}")
    print(f"[vanilla] fwd  rel-err vs recurrent  : {err_vs_rec:.4e}")
    assert err_vs_ref < 0.02, f"Forward error too large: {err_vs_ref}"

    # --- Backward correctness ---
    q, k, v, g, scale = make_vanilla_linear_attn_inputs(
        B, H, T, D, DV, dtype=dtype, device=DEVICE, requires_grad=True
    )
    q_s = q * scale

    out = chunked_linear_attn(q_s, k, v, g, C=C)
    loss = out.sum()
    loss.backward()

    assert q.grad is not None, "q.grad is None"
    assert k.grad is not None, "k.grad is None"
    assert v.grad is not None, "v.grad is None"

    # Reference grads
    q2, k2, v2, g2, _ = make_vanilla_linear_attn_inputs(
        B, H, T, D, DV, dtype=dtype, device=DEVICE, requires_grad=True
    )
    q2.data.copy_(q.data)
    k2.data.copy_(k.data)
    v2.data.copy_(v.data)
    g2 = g.detach().clone()
    q2_s = q2 * scale

    ref_out = chunked_linear_attn_reference(q2_s, k2, v2, g2, C=C)
    ref_out.sum().backward()

    for name, grad, ref_grad in [
        ("q", q.grad, q2.grad),
        ("k", k.grad, k2.grad),
        ("v", v.grad, v2.grad),
    ]:
        err = _rel_error(grad, ref_grad)
        print(f"[vanilla] bwd  grad-{name} rel-err: {err:.4e}")
        assert err < 0.05, f"Backward grad-{name} error too large: {err}"

    print("[vanilla] all tests passed")


def benchmark():
    B, H, T, D, DV, C = 2, 4, 1024, 64, 64, 64
    dtype = torch.bfloat16

    q, k, v, g, scale = make_vanilla_linear_attn_inputs(
        B, H, T, D, DV, dtype=dtype, device=DEVICE
    )
    q_s = q * scale

    fwd_ms = do_bench(lambda: chunked_linear_attn(q_s, k, v, g, C=C))
    print(f"[vanilla] fwd  {fwd_ms:.3f} ms  (B={B}, H={H}, T={T}, D={D}, DV={DV})")

    q, k, v, g, scale = make_vanilla_linear_attn_inputs(
        B, H, T, D, DV, dtype=dtype, device=DEVICE, requires_grad=True
    )
    q_s = q * scale
    out = chunked_linear_attn(q_s, k, v, g, C=C)

    def fwd_bwd():
        out_ = chunked_linear_attn(q_s, k, v, g, C=C)
        out_.sum().backward(retain_graph=True)

    fwd_bwd_ms = do_bench(fwd_bwd)
    print(f"[vanilla] fwd+bwd  {fwd_bwd_ms:.3f} ms")


def main():
    test()
    benchmark()


if __name__ == "__main__":
    main()
