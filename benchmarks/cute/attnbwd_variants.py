"""Attention-backward benchmark variants and the chunked fp32 reference.

Variant axes: head dim (64/128), causal, sequence length, batch/heads, GQA,
dtype. Shapes follow common training configs (tritonbench / flash-attention
benchmark setups); all are compute-bound at these sizes.

``prepare_reference`` computes, in chunked fp32 on-GPU math: the forward
output ``o`` (cast to the input dtype) and base-2 LSE (inputs the Helion
backward consumes, normally produced by the forward kernel), plus reference
gradients dq/dk/dv in fp32 used for every implementation's accuracy check.
"""

from __future__ import annotations

import dataclasses

import torch

RCP_LN2 = 1.4426950408889634


@dataclasses.dataclass(frozen=True)
class Variant:
    name: str
    b: int
    hq: int
    hkv: int
    s: int
    d: int
    causal: bool
    dtype: torch.dtype


VARIANTS: dict[str, Variant] = {
    v.name: v
    for v in (
        Variant("d64", 4, 32, 32, 8192, 64, False, torch.bfloat16),
        Variant("d64-causal", 4, 32, 32, 8192, 64, True, torch.bfloat16),
        Variant("d128", 4, 16, 16, 8192, 128, False, torch.bfloat16),
        Variant("d128-causal", 4, 16, 16, 8192, 128, True, torch.bfloat16),
        Variant("s16k-causal", 2, 16, 16, 16384, 128, True, torch.bfloat16),
        Variant("s4k-b8", 8, 16, 16, 4096, 128, True, torch.bfloat16),
        Variant("gqa-causal", 4, 32, 8, 8192, 128, True, torch.bfloat16),
        Variant("d64-fp16", 4, 32, 32, 8192, 64, False, torch.float16),
    )
}


def variant_flops(variant: Variant) -> float:
    """Backward FLOPs: 5 gemms of D MACs per attended (q, kv) pair."""
    if variant.causal:
        pairs = variant.b * variant.hq * variant.s * (variant.s + 1) / 2
    else:
        pairs = variant.b * variant.hq * variant.s * variant.s
    return 10.0 * pairs * variant.d


def make_inputs(
    variant: Variant, device: torch.device, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q, do = (
        torch.randn(
            (variant.b, variant.hq, variant.s, variant.d),
            dtype=variant.dtype,
            generator=gen,
        ).to(device)
        for _ in range(2)
    )
    k, v = (
        torch.randn(
            (variant.b, variant.hkv, variant.s, variant.d),
            dtype=variant.dtype,
            generator=gen,
        ).to(device)
        for _ in range(2)
    )
    return q, k, v, do


def prepare_reference(
    variant: Variant,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    chunk: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (o [input dtype], lse2 fp32, dq fp32, dk fp32, dv fp32)."""
    b, hq, s, d = q.shape
    hkv = k.shape[1]
    group = hq // hkv
    scale = d**-0.5
    device = q.device
    o = torch.empty_like(q)
    lse2 = torch.empty(b, hq, s, dtype=torch.float32, device=device)
    dq = torch.empty(q.shape, dtype=torch.float32, device=device)
    dk = torch.zeros(k.shape, dtype=torch.float32, device=device)
    dv = torch.zeros(v.shape, dtype=torch.float32, device=device)
    pos = torch.arange(s, device=device)
    for bi in range(b):
        for h in range(hq):
            kh = k[bi, h // group].float()
            vh = v[bi, h // group].float()
            qh = q[bi, h].float()
            doh = do[bi, h].float()
            for i0 in range(0, s, chunk):
                sc = qh[i0 : i0 + chunk] @ kh.T * scale
                if variant.causal:
                    sc = sc.masked_fill(
                        pos[None, :] > pos[i0 : i0 + chunk, None], float("-inf")
                    )
                lse = torch.logsumexp(sc, -1)
                p = torch.exp(sc - lse[:, None])
                o[bi, h, i0 : i0 + chunk] = (p @ vh).to(q.dtype)
                lse2[bi, h, i0 : i0 + chunk] = lse * RCP_LN2
            # delta from the dtype-rounded o, matching what backward kernels see
            oh = o[bi, h].float()
            delta = (doh * oh).sum(-1)
            for i0 in range(0, s, chunk):
                qs = qh[i0 : i0 + chunk]
                dos = doh[i0 : i0 + chunk]
                sc = qs @ kh.T * scale
                if variant.causal:
                    sc = sc.masked_fill(
                        pos[None, :] > pos[i0 : i0 + chunk, None], float("-inf")
                    )
                p = torch.exp(sc - (lse2[bi, h, i0 : i0 + chunk, None] / RCP_LN2))
                dp = dos @ vh.T
                ds = p * (dp - delta[i0 : i0 + chunk, None])
                dq[bi, h, i0 : i0 + chunk] = ds @ kh * scale
                dk[bi, h // group] += ds.T @ qs * scale
                dv[bi, h // group] += p.T @ dos
    return o, lse2, dq, dk, dv


def check_grads(
    variant: Variant,
    got: tuple[torch.Tensor, ...],
    ref: tuple[torch.Tensor, ...],
) -> tuple[str, float]:
    """Max relative-to-peak error across dq/dk/dv vs the fp32 reference."""
    worst = 0.0
    for g, r in zip(got, ref, strict=True):
        rel = (g.float() - r).abs().max().item() / max(r.abs().max().item(), 1e-6)
        worst = max(worst, rel)
    tol = 3e-2 if variant.dtype is torch.bfloat16 else 2e-2
    return ("PASS" if worst < tol else "FAIL"), worst
