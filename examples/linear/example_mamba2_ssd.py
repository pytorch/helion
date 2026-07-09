"""
Mamba-2 SSD Example
====================

Demonstrates chunked linear attention on the Mamba-2 SSD variant using
mamba_ssm as the reference implementation. Includes correctness tests against
a naive recurrent reference, mamba_ssm, and chunked reference backward, plus
a benchmark suite comparing against mamba_ssm.
"""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING
import warnings

import torch
import torch.nn.functional as F
from triton.testing import do_bench

from .linear_attention_engine import LinearAttentionVariant
from .linear_attention_engine import get_helion_fwd_kernel
from .linear_attention_engine import recurrent_step
from .linear_attention_harness import make_mamba2_ssd_inputs
from .linear_attention_utils import chunked_linear_attn_reference
from .linear_attention_utils import head_to_time_first as _htf
from .linear_attention_utils import naive_recurrent_reference
from .linear_attention_utils import rel_error as _rel_error
from helion._testing import DEVICE

if TYPE_CHECKING:
    from collections.abc import Callable

# Test/benchmark config (use T=128 to avoid numerical overflow in cumulative decay)
B, H, T, D, DV = 2, 4, 128, 32, 16
C = 32
DTYPE = torch.bfloat16
BENCH_CONFIGS = [(1, 32, 2048, 128, 128), (1, 32, 4096, 128, 128)]
BENCH_C = 64
HELION_FWD = get_helion_fwd_kernel(LinearAttentionVariant.MAMBA2_SSD)


def _import_mamba() -> Callable[..., torch.Tensor] | None:
    """Import mamba_chunk_scan_combined with shims for missing CUDA extensions."""
    for m in ["selective_scan_cuda", "causal_conv1d_cuda", "causal_conv1d"]:
        if m not in sys.modules:
            sys.modules[m] = types.ModuleType(m)
    try:
        from mamba_ssm.ops.triton.ssd_combined import (  # pyrefly: ignore[missing-import]
            mamba_chunk_scan_combined,
        )
    except ImportError:
        warnings.warn(
            "mamba_ssm not installed, skipping mamba comparisons", stacklevel=2
        )
        return None

    return mamba_chunk_scan_combined


def _make_mamba_native_inputs(
    bi: int,
    hi: int,
    ti: int,
    di: int,
    dvi: int,
    dtype: torch.dtype,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create mamba-style native inputs (time-first layout)."""
    dt = F.softplus(torch.randn(bi, ti, hi, device=device, dtype=dtype))
    A = -torch.rand(hi, device=device, dtype=dtype) - 0.5
    B_mat = torch.randn(bi, ti, hi, di, device=device, dtype=dtype)
    C_mat = torch.randn(bi, ti, hi, di, device=device, dtype=dtype)
    x = torch.randn(bi, ti, hi, dvi, device=device, dtype=dtype)
    return x, dt, A, B_mat, C_mat


def test() -> None:
    """Forward + backward correctness vs reference and mamba_ssm."""
    torch.manual_seed(42)
    mamba_chunk_scan_combined = _import_mamba()
    _has_mamba = mamba_chunk_scan_combined is not None

    # === Make inputs ===
    inputs = make_mamba2_ssd_inputs(B, H, T, D, DV, dtype=DTYPE, device=DEVICE)

    # === Forward: vs naive recurrent reference ===
    out = HELION_FWD(inputs.q, inputs.k, inputs.v, inputs.g, C=C, scale=inputs.scale)
    assert isinstance(out, torch.Tensor)
    assert inputs.g is not None
    ref = naive_recurrent_reference(
        inputs.q, inputs.k, inputs.v, inputs.g, q_scale=inputs.scale
    )
    fwd_err = _rel_error(out, ref)
    assert fwd_err < 0.02, f"Forward error: {fwd_err}"
    print(f"  fwd vs recurrent: {fwd_err:.4e} PASS")

    # === Forward: vs mamba_ssm ===
    # Build mamba-native inputs and the matching q/k/v/g unconditionally so they
    # stay bound for the backward comparison below; only the comparison itself is
    # gated on mamba_ssm being importable.
    x, dt, A, B_mat, C_mat = _make_mamba_native_inputs(B, H, T, D, DV, DTYPE, DEVICE)
    q_m = C_mat.transpose(1, 2).contiguous()
    k_m = B_mat.transpose(1, 2).contiguous()
    v_m = (x * dt.unsqueeze(-1)).transpose(1, 2).contiguous()
    g_m = (A[None, None, :] * dt).transpose(1, 2).contiguous()
    if _has_mamba:
        out_m = HELION_FWD(q_m, k_m, v_m, g_m, C=C, scale=1.0)
        assert isinstance(out_m, torch.Tensor)
        o_mamba = mamba_chunk_scan_combined(
            x, dt, A, B_mat, C_mat, chunk_size=C, D=None, dt_softplus=False
        )
        o_mamba_hf = o_mamba.transpose(1, 2).contiguous()
        fla_err = _rel_error(out_m, o_mamba_hf)
        print(
            f"  fwd vs mamba_ssm: {fla_err:.4e} {'PASS' if fla_err < 0.02 else 'FAIL'}"
        )

    # === Backward: vs chunked reference ===
    grad_out = torch.randn(B, H, T, DV, device=DEVICE, dtype=DTYPE)
    q1 = inputs.q.clone().requires_grad_(True)
    k1 = inputs.k.clone().requires_grad_(True)
    v1 = inputs.v.clone().requires_grad_(True)
    o1 = HELION_FWD(q1, k1, v1, inputs.g, C=C, scale=inputs.scale)
    assert isinstance(o1, torch.Tensor)
    o1.backward(grad_out)

    q2 = inputs.q.clone().requires_grad_(True)
    k2 = inputs.k.clone().requires_grad_(True)
    v2 = inputs.v.clone().requires_grad_(True)
    o2 = chunked_linear_attn_reference(q2, k2, v2, inputs.g, C=C)
    o2.backward(grad_out)

    for name, g1, g2 in [
        ("dq", q1.grad, q2.grad),
        ("dk", k1.grad, k2.grad),
        ("dv", v1.grad, v2.grad),
    ]:
        err = _rel_error(g1, g2)
        assert err < 0.05, f"Backward {name} error: {err}"
        print(f"  bwd {name} vs ref: {err:.4e} PASS")

    # === Backward: vs mamba_ssm (dq/dC comparison) ===
    if _has_mamba:
        q3 = q_m.clone().requires_grad_(True)
        k3 = k_m.clone().requires_grad_(True)
        v3 = v_m.clone().requires_grad_(True)
        o3 = HELION_FWD(q3, k3, v3, g_m, C=C, scale=1.0)
        assert isinstance(o3, torch.Tensor)
        grad_out_m = torch.randn(B, H, T, DV, device=DEVICE, dtype=DTYPE)
        o3.backward(grad_out_m)

        C2 = C_mat.clone().requires_grad_(True)
        B2 = B_mat.clone().requires_grad_(True)
        x2 = x.clone().requires_grad_(True)
        dt2 = dt.clone().requires_grad_(True)
        o4 = mamba_chunk_scan_combined(
            x2, dt2, A, B2, C2, chunk_size=C, D=None, dt_softplus=False
        )
        o4.backward(_htf(grad_out_m))

        assert C2.grad is not None and B2.grad is not None
        dq_err = _rel_error(q3.grad, C2.grad.transpose(1, 2).contiguous())
        print(f"  bwd dq vs mamba:  {dq_err:.4e} {'PASS' if dq_err < 0.05 else 'FAIL'}")
        dk_err = _rel_error(k3.grad, B2.grad.transpose(1, 2).contiguous())
        print(f"  bwd dk vs mamba:  {dk_err:.4e} (info)")

    # === Recurrent step: compare step-by-step vs chunked ===
    torch.manual_seed(42)
    rec_inputs = make_mamba2_ssd_inputs(B, H, T, D, DV, dtype=DTYPE, device=DEVICE)
    assert rec_inputs.g is not None

    o_chunked = HELION_FWD(
        rec_inputs.q,
        rec_inputs.k,
        rec_inputs.v,
        rec_inputs.g,
        C=C,
        scale=rec_inputs.scale,
    )
    assert isinstance(o_chunked, torch.Tensor)

    state = torch.zeros(B, H, D, DV, device=DEVICE, dtype=torch.float32)
    o_steps = []
    for t in range(T):
        alpha = torch.exp(rec_inputs.g[:, :, t : t + 1])  # [B,H,1]
        o_t, state = recurrent_step(
            rec_inputs.q[:, :, t : t + 1],
            rec_inputs.k[:, :, t : t + 1],
            rec_inputs.v[:, :, t : t + 1],
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
    """Benchmark forward and fwd+bwd, comparing against mamba_ssm."""
    mamba_chunk_scan_combined = _import_mamba()
    if mamba_chunk_scan_combined is None:
        warnings.warn("mamba_ssm not installed, skipping benchmark", stacklevel=1)
        return

    print(
        f"{'Config':<24} {'Helion fwd':>10} {'Mamba fwd':>10}"
        f" {'Helion f+b':>12} {'Mamba f+b':>12}"
    )
    print("-" * 72)

    for bi, hi, ti, di, dvi in BENCH_CONFIGS:
        inputs = make_mamba2_ssd_inputs(
            bi, hi, ti, di, dvi, dtype=DTYPE, device=DEVICE, requires_grad=True
        )
        grad_out = torch.randn(bi, hi, ti, dvi, device=DEVICE, dtype=DTYPE)

        # Mamba native inputs (with grad for backward benchmark)
        x, dt, A, B_mat, C_mat = _make_mamba_native_inputs(
            bi, hi, ti, di, dvi, DTYPE, DEVICE
        )
        x = x.requires_grad_(True)
        dt = dt.requires_grad_(True)
        B_mat = B_mat.requires_grad_(True)
        C_mat = C_mat.requires_grad_(True)
        go_t = _htf(grad_out)

        def helion_fwd(
            qi: torch.Tensor = inputs.q,
            ki: torch.Tensor = inputs.k,
            vi: torch.Tensor = inputs.v,
            gi: torch.Tensor | None = inputs.g,
            scale: float = inputs.scale,
        ) -> torch.Tensor:
            o = HELION_FWD(qi, ki, vi, gi, C=BENCH_C, scale=scale)
            assert isinstance(o, torch.Tensor)
            return o

        fwd_ms = do_bench(helion_fwd)

        def mamba_fwd(
            x: torch.Tensor = x,
            dt: torch.Tensor = dt,
            A: torch.Tensor = A,
            Bm: torch.Tensor = B_mat,
            Cm: torch.Tensor = C_mat,
            _fn: Callable[..., torch.Tensor] = mamba_chunk_scan_combined,
        ) -> torch.Tensor:
            return _fn(x, dt, A, Bm, Cm, chunk_size=BENCH_C, D=None, dt_softplus=False)

        mamba_fwd_ms = do_bench(mamba_fwd)

        def helion_fb(
            qi: torch.Tensor = inputs.q,
            ki: torch.Tensor = inputs.k,
            vi: torch.Tensor = inputs.v,
            gi: torch.Tensor | None = inputs.g,
            scale: float = inputs.scale,
            go: torch.Tensor = grad_out,
        ) -> None:
            o = HELION_FWD(qi, ki, vi, gi, C=BENCH_C, scale=scale)
            assert isinstance(o, torch.Tensor)
            o.backward(go)
            qi.grad = ki.grad = vi.grad = None

        fb_ms = do_bench(helion_fb)

        def mamba_fb(
            x: torch.Tensor = x,
            dt: torch.Tensor = dt,
            A: torch.Tensor = A,
            Bm: torch.Tensor = B_mat,
            Cm: torch.Tensor = C_mat,
            go: torch.Tensor = go_t,
            _fn: Callable[..., torch.Tensor] = mamba_chunk_scan_combined,
        ) -> None:
            o = _fn(x, dt, A, Bm, Cm, chunk_size=BENCH_C, D=None, dt_softplus=False)
            o.backward(go)
            x.grad = dt.grad = Bm.grad = Cm.grad = None

        mamba_fb_ms = do_bench(mamba_fb)

        cfg = f"({bi},{hi},{ti},{di},{dvi})"
        print(
            f"{cfg:<24} {fwd_ms:>10.3f} {mamba_fwd_ms:>10.3f}"
            f" {fb_ms:>12.3f} {mamba_fb_ms:>12.3f}"
        )


def main() -> None:
    print("=== Mamba-2 SSD ===")
    test()
    print()
    benchmark()


if __name__ == "__main__":
    main()
