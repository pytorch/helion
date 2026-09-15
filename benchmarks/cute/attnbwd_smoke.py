"""Small-shape correctness smoke test for the attention-backward kernels.

Usage: python benchmarks/cute/attnbwd_smoke.py [case ...]
Honors HELION_BACKEND. Pins block_sizes=[64, 64] on the main kernels and
[64] on the preprocess kernel rather than relying on autotuning.
"""

from __future__ import annotations

import os
import sys

from attnbwd_kernels import RCP_LN2  # pyrefly: ignore [missing-import]
from attnbwd_kernels import attention_bwd  # pyrefly: ignore [missing-import]
from attnbwd_kernels import attention_bwd_causal  # pyrefly: ignore [missing-import]
from attnbwd_kernels import attention_bwd_dkdv  # pyrefly: ignore [missing-import]
from attnbwd_kernels import (  # pyrefly: ignore [missing-import]
    attention_bwd_dkdv_causal,
)
from attnbwd_kernels import attention_bwd_dq  # pyrefly: ignore [missing-import]
from attnbwd_kernels import attention_bwd_dq_causal  # pyrefly: ignore [missing-import]
from attnbwd_kernels import attention_bwd_preprocess  # pyrefly: ignore [missing-import]
import torch

from helion._testing import code_and_output

CASES = {
    "noncausal": (2, 4, 4, 256, 64, False),
    "causal": (2, 4, 4, 256, 64, True),
    "gqa": (2, 4, 2, 256, 64, False),
    "gqa-causal": (2, 4, 2, 256, 64, True),
    "noncausal-d128": (2, 2, 2, 256, 128, False),
}


def ref_grads(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    sm_scale: float,
    causal: bool,
) -> tuple[torch.Tensor, ...]:
    group = q.size(1) // k.size(1)
    q32 = q.detach().float().requires_grad_()
    k32 = k.detach().float().requires_grad_()
    v32 = v.detach().float().requires_grad_()
    ke = k32.repeat_interleave(group, dim=1)
    ve = v32.repeat_interleave(group, dim=1)
    scores = q32 @ ke.transpose(-1, -2) * sm_scale
    if causal:
        m, n = scores.shape[-2:]
        keep = torch.ones(m, n, dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~keep, float("-inf"))
    p = scores.softmax(-1)
    out = p @ ve
    out.backward(do.float())
    lse2 = torch.logsumexp(scores, dim=-1) * RCP_LN2
    assert q32.grad is not None and k32.grad is not None and v32.grad is not None
    return out.detach(), lse2, q32.grad, k32.grad, v32.grad


def smoke(name: str, mode: str = "split") -> bool:
    b, hq, hkv, s, d, causal = CASES[name]
    group = hq // hkv
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(0)
    q, do = [torch.randn(b, hq, s, d, device=device, dtype=dtype) for _ in range(2)]
    k, v = [torch.randn(b, hkv, s, d, device=device, dtype=dtype) for _ in range(2)]
    sm_scale = d**-0.5
    out32, lse2, dq_ref, dk_ref, dv_ref = ref_grads(q, k, v, do, sm_scale, causal)
    o = out32.to(dtype)

    _, delta = code_and_output(attention_bwd_preprocess, (o, do), block_sizes=[64])
    kg = k.reshape(b * hkv, s, d)
    vg = v.reshape(b * hkv, s, d)
    if causal:
        args = (
            q.reshape(b * hkv, group, s, d),
            kg,
            vg,
            lse2.reshape(b * hkv, group, s),
            do.reshape(b * hkv, group, s, d),
            delta.reshape(b * hkv, group, s),
        )
        if mode == "split":
            code, (dk, dv) = code_and_output(
                attention_bwd_dkdv_causal, (*args, sm_scale), block_sizes=[64, 64]
            )
            code2, dq = code_and_output(
                attention_bwd_dq_causal, (*args, sm_scale), block_sizes=[64, 64]
            )
            code += code2
        else:
            code, (dq, dk, dv) = code_and_output(
                attention_bwd_causal, (*args, sm_scale), block_sizes=[64, 64]
            )
    else:
        args = (
            q.reshape(b * hkv, group * s, d),
            kg,
            vg,
            lse2.reshape(b * hkv, group * s),
            do.reshape(b * hkv, group * s, d),
            delta.reshape(b * hkv, group * s),
        )
        if mode == "split":
            code, (dk, dv) = code_and_output(
                attention_bwd_dkdv, (*args, sm_scale), block_sizes=[64, 64]
            )
            code2, dq = code_and_output(
                attention_bwd_dq, (*args, sm_scale), block_sizes=[64, 64]
            )
            code += code2
        else:
            code, (dq, dk, dv) = code_and_output(
                attention_bwd, (*args, sm_scale), block_sizes=[64, 64]
            )

    results = []
    for label, got, ref in (
        ("dq", dq.reshape(q.shape).float(), dq_ref),
        ("dk", dk.reshape(k.shape).float(), dk_ref),
        ("dv", dv.reshape(v.shape).float(), dv_ref),
    ):
        rel = (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)
        results.append((label, rel))
    ok = all(rel < 3e-2 for _, rel in results)
    marker = ""
    if os.environ.get("HELION_BACKEND") == "cute":
        marker = " flash" if "flash" in code else " generic"
    detail = " ".join(f"{label}={rel:.5f}" for label, rel in results)
    print(f"{name:16s} {mode:5s} {detail} {'PASS' if ok else 'FAIL'}{marker}")
    return ok


def main() -> None:
    modes = os.environ.get("ATTNBWD_SMOKE_MODES", "split,fused").split(",")
    names = sys.argv[1:] or list(CASES)
    failures = []
    for name in names:
        for mode in modes:
            try:
                ok = smoke(name, mode)
            except Exception as e:
                print(f"{name:16s} {mode:5s} ERROR {type(e).__name__}: {str(e)[:200]}")
                ok = False
            if not ok:
                failures.append(f"{name}/{mode}")
    if failures:
        print("FAILURES:", ", ".join(failures))
        sys.exit(1)
    print("all pass")


if __name__ == "__main__":
    main()
