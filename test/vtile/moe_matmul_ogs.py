from __future__ import annotations

import os

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=False, autotune_effort="none")
def moe_matmul_ogs_vtile_nomask(
    a: torch.Tensor,
    w: torch.Tensor,
    expert_token_counts: torch.Tensor,
    expert_token_offsets: torch.Tensor,
    sorted_to_orig_token_idx: torch.Tensor,
    max_t_per_expert: int,
) -> torch.Tensor:
    del max_t_per_expert
    t, k = a.shape
    e, _, n = w.shape
    c = torch.zeros(
        t,
        n,
        dtype=torch.promote_types(a.dtype, w.dtype),
        device=a.device,
    )

    for tile_e in hl.tile(e):
        starts = expert_token_offsets[tile_e]
        counts = expert_token_counts[tile_e]

        for tile_t in hl.vtile(counts):
            sorted_idx = starts[:, None] + tile_t.index
            orig_idx = sorted_to_orig_token_idx[sorted_idx]
            for tile_n in hl.tile(n):
                acc = hl.zeros([tile_e, tile_t, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    a_frag = a[orig_idx, tile_k]
                    w_frag = w[tile_e, tile_k, tile_n]
                    acc = acc + torch.matmul(a_frag, w_frag)
                c[orig_idx, tile_n] = acc.to(c.dtype)

    return c


def moe_matmul_ogs_kernel_args_gen(
    a: torch.Tensor,
    w: torch.Tensor,
    top1_expert_per_token: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    e = w.size(0)
    device = a.device
    sorted_to_orig_token_idx = torch.argsort(top1_expert_per_token, stable=True).to(
        torch.int32
    )
    expert_token_counts = torch.bincount(top1_expert_per_token, minlength=e).to(
        torch.int32
    )
    expert_token_offsets = torch.empty(e + 1, dtype=torch.int32, device=device)
    expert_token_offsets[0] = 0
    expert_token_offsets[1:] = torch.cumsum(expert_token_counts, 0, dtype=torch.int32)
    max_t_per_expert = int(expert_token_counts.max().item())
    return (
        a,
        w,
        expert_token_counts,
        expert_token_offsets,
        sorted_to_orig_token_idx,
        max_t_per_expert,
    )


def reference_moe_matmul_ogs(
    a: torch.Tensor, w: torch.Tensor, top1_expert_per_token: torch.Tensor
) -> torch.Tensor:
    t, _ = a.shape
    n = w.size(2)
    dtype = torch.promote_types(a.dtype, w.dtype)
    c = torch.empty(t, n, device=a.device, dtype=dtype)
    n_experts = w.size(0)
    for e in range(n_experts):
        token_idx = (top1_expert_per_token == e).nonzero(as_tuple=True)[0]
        if token_idx.numel() == 0:
            continue
        c[token_idx] = a[token_idx] @ w[e]
    return c


def main() -> None:
    torch.manual_seed(0)
    device = "cpu"
    dtype = torch.float32
    t, k, n, n_experts = 256, 96, 64, 8

    a = torch.randn(t, k, device=device, dtype=dtype)
    w = torch.randn(n_experts, k, n, device=device, dtype=dtype)
    top1_expert_per_token = torch.randint(n_experts, (t,), device=device)
    kernel_args = moe_matmul_ogs_kernel_args_gen(a, w, top1_expert_per_token)

    out = moe_matmul_ogs_vtile_nomask(*kernel_args)
    ref = reference_moe_matmul_ogs(a, w, top1_expert_per_token)
    max_diff = (out - ref).abs().max().item()
    print(f"max_diff={max_diff}")

    if os.environ.get("HELION_PRINT_OUTPUT_CODE") == "1":
        print("Output code printed by Helion (see stderr).")


if __name__ == "__main__":
    main()
