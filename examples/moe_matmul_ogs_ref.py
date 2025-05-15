#!/usr/bin/env python3
# moe_gather_scatter_ref_multiW.py
#
# Gather-GEMM-scatter reference where *each expert owns its own W*.

import torch
torch.manual_seed(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------------------------------------------------------------
# Problem sizes
# ---------------------------------------------------------------------
B        = 1024               # tokens / rows  (multiple of BLOCK_M)
K        = 512                # hidden size
N        = 256                # output size
BLOCK_M  = 16                 # rows per micro-tile
n_experts = 32                # number of experts
dtype    = torch.float16

assert B % BLOCK_M == 0, "B must be a multiple of BLOCK_M."

# ---------------------------------------------------------------------
# Top-1 gating  → expert-sorted permutation
# ---------------------------------------------------------------------
top1_expert  = torch.randint(n_experts, (B,), device=device)
sorting      = torch.argsort(top1_expert)            # gather permutation
inverse_perm = torch.empty_like(sorting)
inverse_perm[sorting] = torch.arange(B, device=device)

# ---------------------------------------------------------------------
# Matrices
# ---------------------------------------------------------------------
A = torch.randn(B, K, device=device, dtype=dtype)
# one weight matrix per expert
W = torch.randn(n_experts, K, N, device=device, dtype=dtype)

# ---------------------------------------------------------------------
# 1) Golden reference – loop over experts
# ---------------------------------------------------------------------
C_ref = torch.empty(B, N, device=device, dtype=dtype)

for e in range(n_experts):
    token_idx = (top1_expert == e).nonzero(as_tuple=True)[0]   # rows for expert e
    if token_idx.numel() == 0:
        continue
    C_ref[token_idx] = A[token_idx] @ W[e]                    # [Ne, K]·[K, N]

# ---------------------------------------------------------------------
# 2) Indirect path mimicking Triton matmul_ogs
#    Work in expert-sorted order, then unsort once.
# ---------------------------------------------------------------------
A_sorted      = A[sorting]                   # [B, K] contiguous in expert order
top1_sorted   = top1_expert[sorting]
C_sorted      = torch.empty(B, N, device=device, dtype=dtype)

# Pre-compute how many tokens each expert owns and where its block starts
counts   = torch.bincount(top1_sorted, minlength=n_experts)        # [E]
starts   = torch.cumsum(counts, 0) - counts                         # prefix sum

for e in range(n_experts):
    length = counts[e].item()
    if length == 0:
        continue
    base   = starts[e].item()
    seg    = slice(base, base + length)         # contiguous segment for this expert

    # Process the segment in BLOCK_M-sized tiles (mirrors kernel behaviour)
    for off in range(base, base + length, BLOCK_M):
        sub = slice(off, off + BLOCK_M)
        a_tile = A_sorted[sub, :]               # [16, K]
        c_tile = a_tile @ W[e]                  # [16, N]
        C_sorted[sub, :] = c_tile

# Scatter (unsort) once
C_indirect = torch.empty_like(C_sorted)
C_indirect[sorting] = C_sorted

# ---------------------------------------------------------------------
# Check correctness
# ---------------------------------------------------------------------
diff = (C_ref - C_indirect).abs().max().item()
print(f"max |Δ| between reference and indirect = {diff:.4e}")
assert diff < 1e-2, "Results differ!  Something is wrong."
print("✓  Per-expert version matches the plain reference.")