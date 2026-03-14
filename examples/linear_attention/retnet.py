"""
RetNet / Linear Attention using the Generic Chunkwise Primitive
================================================================

RetNet's chunk_fwd_h is a special case — no delta correction,
just direct k^T @ v updates with exponential decay.

    kernel = linear_attention_fwd_h(chunk_size=64)

Reference: Sun et al. "Retentive Network" (2023)
"""

from __future__ import annotations

import torch
from chunk_recurrence import linear_attention_fwd_h


def reference_retnet_fwd_h(k, v, g, chunk_size=64):
    B, T, H, K = k.shape
    V = v.shape[-1]
    C = chunk_size
    NT = T // C

    k_c = k.float().reshape(B, NT, C, H, K).permute(0, 1, 3, 2, 4)
    v_c = v.float().reshape(B, NT, C, H, V).permute(0, 1, 3, 2, 4)
    g_c = g.float().reshape(B, NT, C, H).permute(0, 1, 3, 2)

    h_all = torch.zeros(B, NT, H, K, V, dtype=torch.float32, device=k.device)
    h = torch.zeros(B, H, K, V, dtype=torch.float32, device=k.device)

    for c in range(NT):
        h_all[:, c] = h
        g_last = g_c[:, c, :, -1]
        gate = torch.exp(g_last.unsqueeze(-1) - g_c[:, c])
        v_gated = v_c[:, c] * gate.unsqueeze(-1)
        h = h * torch.exp(g_last).unsqueeze(-1).unsqueeze(-1)
        h = h + k_c[:, c].transpose(-1, -2) @ v_gated

    return h_all, v


def check(B, T, H, K, V, chunk_size=64, device="cuda"):
    torch.manual_seed(42)
    k = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    v = torch.randn(B, T, H, V, dtype=torch.float32, device=device)
    w = torch.zeros_like(k)
    g = torch.cumsum(
        -torch.abs(torch.randn(B, T, H, dtype=torch.float32, device=device)),
        dim=1,
    )

    h_ref, _ = reference_retnet_fwd_h(k, v, g, chunk_size)

    kernel = linear_attention_fwd_h(chunk_size=chunk_size)
    h_helion, _ = kernel(k, w, v, g)

    h_ok = torch.allclose(h_helion.float(), h_ref.float(), rtol=1e-2, atol=1e-2)
    print(f"B={B} T={T} H={H} K={K} V={V}: h={'PASS' if h_ok else 'FAIL'}")
    assert h_ok


def main():
    check(2, 128, 4, 64, 64)
    check(1, 256, 2, 64, 128)
    print("All RetNet tests passed!")


if __name__ == "__main__":
    main()
