"""gdn_fwd_h: gated delta-net forward hidden states, one per 64-step chunk.

For each (batch, head), with state ``S`` [K, V] in fp32 starting at zero and
per chunk ``t``: emit ``h_t = bf16(S)``; ``v = u_t - w_t @ bf16(S)``; gate
``v *= exp(g_last - g)``; decay ``S *= exp(g_last)``; ``S += k_t^T @ bf16(v)``.
The bf16 roundings are part of the operator, so the reference keeps them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from helion._testing import import_example

_examples_gdn_fwd_h = import_example("gdn_fwd_h")
helion_gdn_fwd_h = _examples_gdn_fwd_h.helion_gdn_fwd_h
ref_gdn_fwd_h = _examples_gdn_fwd_h.ref_gdn_fwd_h

if TYPE_CHECKING:
    import mlx.core as mx

    from ..manifest import Case

MODE = "fwd"
HELION_KERNELS = (helion_gdn_fwd_h,)
CHUNK = 64


def inputs(case: Case, gen: torch.Generator) -> dict[str, torch.Tensor]:
    b, h, t, k, v = (case.params[key] for key in ("B", "H", "T", "K", "V"))
    nt = t // CHUNK
    # w: each (chunk, head) block is the orthogonal polar factor of a random
    # [64, K] matrix, as TritonBench builds it.
    raw = torch.randn(b, nt, CHUNK, h, k, generator=gen).permute(0, 1, 3, 2, 4)
    wu, _, wv = torch.linalg.svd(raw, full_matrices=False)
    w = (wu @ wv).permute(0, 1, 3, 2, 4).reshape(b, t, h, k)
    increments = (
        0.5 * torch.log(torch.tensor(1 / 256)) * torch.rand(b, t, h, generator=gen)
    )
    return {
        "k": torch.randn(b, t, h, k, generator=gen, dtype=torch.bfloat16),
        "w": w.to(torch.bfloat16),
        "u": torch.randn(b, t, h, v, generator=gen, dtype=torch.bfloat16),
        "g": torch.cumsum(increments, dim=1),
    }


def reference(
    k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor
) -> torch.Tensor:
    b, t, h, dk = k.shape
    dv = u.shape[-1]
    nt = t // CHUNK

    def chunks(x: torch.Tensor) -> torch.Tensor:
        # [B, T, H, D] -> [B, H, NT, CHUNK, D]
        return x.reshape(b, nt, CHUNK, h, -1).permute(0, 3, 1, 2, 4).to(torch.float64)

    kc, wc, uc = chunks(k), chunks(w), chunks(u)
    gc = g.reshape(b, nt, CHUNK, h).permute(0, 3, 1, 2).to(torch.float64)
    g_last = gc[..., -1]
    gate = torch.exp(g_last[..., None] - gc)[..., None]
    decay = torch.exp(g_last)[..., None, None]
    out = torch.empty(b, nt, h, dk, dv, dtype=k.dtype, device=k.device)
    state = torch.zeros(b, h, dk, dv, dtype=torch.float64, device=k.device)
    for i in range(nt):
        rounded = state.to(k.dtype)
        out[:, i] = rounded
        new_v = (uc[:, :, i] - wc[:, :, i] @ rounded.to(torch.float64)) * gate[:, :, i]
        state = state * decay[:, :, i] + kc[:, :, i].transpose(-1, -2) @ new_v.to(
            k.dtype
        ).to(torch.float64)
    return out


def eager(
    k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor
) -> torch.Tensor:
    return ref_gdn_fwd_h(k, w, u, g, CHUNK)


def helion(
    k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor
) -> torch.Tensor:
    return helion_gdn_fwd_h(k, w, u, g, CHUNK)


def mlx(k: mx.array, w: mx.array, u: mx.array, g: mx.array) -> mx.array:
    import mlx.core as mx

    b, t, h, dk = k.shape
    nt = t // CHUNK

    def chunks(x: mx.array) -> mx.array:
        return (
            x.reshape(b, nt, CHUNK, h, -1).transpose(0, 3, 1, 2, 4).astype(mx.float32)
        )

    kc, wc, uc = chunks(k), chunks(w), chunks(u)
    gc = g.reshape(b, nt, CHUNK, h).transpose(0, 3, 1, 2)
    g_last = gc[..., -1]
    gate = mx.exp(g_last[..., None] - gc)[..., None]
    decay = mx.exp(g_last)[..., None, None]
    state = mx.zeros((b, h, dk, u.shape[-1]), dtype=mx.float32)
    out = []
    for i in range(nt):
        rounded = state.astype(k.dtype)
        out.append(rounded)
        new_v = (uc[:, :, i] - wc[:, :, i] @ rounded.astype(mx.float32)) * gate[:, :, i]
        state = state * decay[:, :, i] + kc[:, :, i].swapaxes(-1, -2) @ new_v.astype(
            k.dtype
        ).astype(mx.float32)
    return mx.stack(out, axis=1)
