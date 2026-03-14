"""
DeltaNet (Ungated) using the Generic Chunkwise Primitive
=========================================================

DeltaNet uses delta rule correction but NO gating/decay.

    kernel = delta_net_fwd_h(chunk_size=64)

Reference: Yang et al. "Parallelizing Linear Transformers with the Delta Rule" (NeurIPS 2024)
"""

from __future__ import annotations

import torch
from chunk_recurrence import delta_net_fwd_h


def reference_delta_net_fwd_h(k, w, u, chunk_size=64):
    B, T, H, K = k.shape
    V = u.shape[-1]
    C = chunk_size
    NT = T // C

    k_c = k.float().reshape(B, NT, C, H, K).permute(0, 1, 3, 2, 4)
    w_c = w.float().reshape(B, NT, C, H, K).permute(0, 1, 3, 2, 4)
    u_c = u.float().reshape(B, NT, C, H, V).permute(0, 1, 3, 2, 4)

    h_all = torch.zeros(B, NT, H, K, V, dtype=torch.float32, device=k.device)
    v_new_c = torch.zeros_like(u_c)
    h = torch.zeros(B, H, K, V, dtype=torch.float32, device=k.device)

    for c in range(NT):
        h_all[:, c] = h
        v_new_c[:, c] = u_c[:, c] - w_c[:, c] @ h
        h = h + k_c[:, c].transpose(-1, -2) @ v_new_c[:, c]

    v_new_out = v_new_c.permute(0, 1, 3, 2, 4).reshape(B, T, H, V)
    return h_all, v_new_out


def check(B, T, H, K, V, chunk_size=64, device="cuda"):
    torch.manual_seed(42)
    k = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    w = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    u = torch.randn(B, T, H, V, dtype=torch.float32, device=device)
    g = torch.zeros(B, T, H, dtype=torch.float32, device=device)

    h_ref, v_ref = reference_delta_net_fwd_h(k, w, u, chunk_size)

    kernel = delta_net_fwd_h(chunk_size=chunk_size)
    h_helion, v_helion = kernel(k, w, u, g)

    h_ok = torch.allclose(h_helion.float(), h_ref.float(), rtol=1e-2, atol=1e-2)
    v_ok = torch.allclose(v_helion.float(), v_ref.float(), rtol=1e-2, atol=1e-2)
    print(f"B={B} T={T} H={H} K={K} V={V}: h={'PASS' if h_ok else 'FAIL'}, v={'PASS' if v_ok else 'FAIL'}")
    assert h_ok and v_ok


def main():
    check(2, 128, 4, 64, 64)
    check(1, 256, 2, 64, 128)
    print("All DeltaNet tests passed!")


if __name__ == "__main__":
    main()
