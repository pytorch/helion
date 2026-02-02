# Gated DeltaNet (Gated Delta Rule) Analysis

---
## **MATHEMATICAL CORE**

> **The gated delta rule - error-correcting memory update with decay:**
>
> ```
> h_t = exp(g_t) * h_{t-1} + k_t ⊗ (β_t * (v_t - h_{t-1}^T @ k_t))
> o_t = h_t @ q_t
> ```
>
> Key components:
> - `exp(g_t)`: Decay gate for selective forgetting
> - `v_t - h_{t-1}^T @ k_t`: **The delta/error term** - difference between target value and what memory would predict
> - `β_t`: Update strength gate
> - `k_t ⊗ (...)`: Outer product update to the hidden state matrix
>
> This enables **memory modification** (not just addition) by correcting retrieval errors.

---

## Overview

Gated DeltaNet is a **linear recurrent attention** mechanism that combines two complementary innovations:

1. **Delta Rule**: Precise memory modification through error-correcting updates
2. **Gating**: Adaptive memory control for selective forgetting

Published at [ICLR 2025](https://arxiv.org/abs/2412.06464), Gated DeltaNet consistently outperforms Mamba2 and DeltaNet across multiple benchmarks. It has been adopted as the linear component in **Qwen3-Next**.

## Core Algorithm

### The Problem with Linear Attention

Vanilla linear attention suffers from **memory overload**: the model can only add new key-value associations without erasing existing information. As sequences grow, "retrieval errors" accumulate and degrade performance.

### The Delta Rule Solution

Instead of simply adding new associations:
```
h_t = h_{t-1} + k_t ⊗ v_t  # Standard linear attention (additive)
```

The delta rule performs error-correcting updates:
```
v̂_t = v_t - h_{t-1}^T @ k_t     # Compute error/residual
h_t = h_{t-1} + k_t ⊗ (β_t * v̂_t)  # Update with correction
```

This allows the model to **modify** existing associations, not just add new ones.

### Adding Gating for Forgetting

Gated DeltaNet adds a decay gate to enable selective forgetting:
```
h_t = g_t * h_{t-1} + k_t ⊗ (β_t * v̂_t)
```

Where `g_t = exp(log_g_t)` is a learnable decay gate in log-space for numerical stability.

## Mathematical Formulation

### Full Update Equation

For each timestep t:
```python
# 1. Apply decay (forgetting)
h_t = exp(g_t) * h_{t-1}

# 2. Compute error/residual
v_residual = v_t - (h_t @ k_t)  # What the memory would predict

# 3. Apply update with beta gate
v_corrected = beta_t * v_residual

# 4. Update memory
h_t = h_t + outer(k_t, v_corrected)

# 5. Compute output
o_t = h_t @ q_t
```

### State Dimensions

- **Query/Key**: `[B, T, H, K]` where K is typically 512
- **Value**: `[B, T, H, V]` where V is typically 512
- **Hidden State**: `[B, H, K, V]` - a matrix per head
- **Gates**: `g` is `[B, T, H]`, `beta` is `[B, T, H]`

## Kernel Variants in This Directory

### 1. `naive.py` - Reference Implementation

The simplest implementation for correctness verification:

```python
def naive_recurrent_gated_delta_rule(q, k, v, beta, g, ...):
    for i in range(T):
        # Apply decay gate
        h = h * g[:, :, i].exp()[..., None, None]

        # Delta rule update
        b_v = v[:, :, i] - (h * k[:, :, i][..., None]).sum(-2)  # Residual
        b_v = b_v * beta[:, :, i][..., None]                     # Gate
        h = h + k[:, :, i].unsqueeze(-1) * b_v.unsqueeze(-2)     # Update

        # Memory readout
        o[:, :, i] = torch.einsum('bhd,bhdm->bhm', q[:, :, i], h)
```

**Complexity**: O(T * K * V) per sequence - sequential, not parallelizable.

### 2. `fused_recurrent.py` - Fused Triton Kernel

**Purpose**: Fused recurrent computation for inference (forward only).

```python
@triton.jit
def fused_recurrent_gated_delta_rule_fwd_kernel(
    q, k, v, g, gk, gv, beta, o, h0, ht, ...
):
    # Initialize hidden state
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        b_h += tl.load(p_h0, ...)

    for _ in range(0, T):
        b_q = tl.load(p_q, ...) * scale
        b_k = tl.load(p_k, ...)
        b_v = tl.load(p_v, ...)
        b_beta = tl.load(p_beta, ...)

        # Apply decay gates
        if USE_G:
            b_h *= exp(tl.load(p_g))
        if USE_GK:
            b_h *= exp(tl.load(p_gk)[:, None])  # Per-key decay
        if USE_GV:
            b_h *= exp(tl.load(p_gv)[None, :])  # Per-value decay

        # Delta rule update
        b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
        b_h += b_k[:, None] * b_v

        # Output
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o)
```

**Features**:
- Three gating options: `g` (scalar), `gk` (per-key), `gv` (per-value)
- Optional L2 normalization of Q/K in kernel
- Variable-length sequence support via `cu_seqlens`
- GVA (Grouped Value Attention) support: `HV > H`

**Grid**: `(NV, N * HV)` - parallelizes over value blocks and batch*heads

### 3. `chunk.py` - Chunk-Parallel Algorithm

**Purpose**: Training-efficient implementation using chunk-level parallelism.

The key insight is to decompose the recurrence into:
1. **Intra-chunk**: Process tokens within each chunk
2. **Inter-chunk**: Propagate states between chunks

```python
def chunk_gated_delta_rule_fwd(q, k, v, g, beta, scale, initial_state, ...):
    # 1. Cumulative sum of gates within chunks
    g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens)

    # 2. Compute WY representation (efficient delta rule encoding)
    A = chunk_scaled_dot_kkt_fwd(k=k, g=g, beta=beta, ...)
    A = solve_tril(A=A, ...)  # Lower triangular solve

    # 3. Recompute w, u from WY representation
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g=g, ...)

    # 4. Chunk-level hidden state propagation
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=g, initial_state=initial_state, ...
    )

    # 5. Compute output
    o = chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=g, scale=scale, ...)

    return g, o, A, final_state, initial_state
```

**WY Representation**: Transforms the recurrence into a form amenable to parallel computation using matrix factorization.

### 4. `wy_fast.py` / `wy_fast_nvlabs.py` - WY Transform Kernels

Efficient Triton kernels for computing the WY representation used in chunk-parallel training.

### 5. `chunk_nvlabs.py` - NVLabs Optimized Version

NVIDIA-optimized chunk implementation, likely with hardware-specific tuning.

## Comparison of Implementations

| Implementation | Use Case | Parallelism | Training | Inference |
|---------------|----------|-------------|----------|-----------|
| `naive.py` | Testing | None | ✓ | ✓ |
| `fused_recurrent.py` | Inference | Per-head | ✗ | ✓ |
| `chunk.py` | Training | Chunk-level | ✓ | ✓ |
| `chunk_nvlabs.py` | Training | Chunk-level | ✓ | ✓ |

## Performance Characteristics

### Complexity

- **Time**: O(T * K * V) total, but chunk algorithm allows O(T/C) parallel chunks
- **Memory**: O(K * V) per head for hidden state (much smaller than O(T) for attention)
- **Chunk Size**: Typically 64 tokens for good parallelism/accuracy tradeoff

### Benchmarks (from paper)

Gated DeltaNet outperforms:
- **Mamba2** on language modeling, commonsense reasoning
- **DeltaNet** on in-context retrieval tasks
- Enables **length extrapolation** beyond training context

## Use Cases

1. **Long-context language modeling**: Constant memory per token
2. **Efficient inference**: State can be cached and updated incrementally
3. **Hybrid architectures**: Combine with sliding window attention (as in Qwen3-Next)
4. **Streaming applications**: Process tokens one at a time with fixed memory

## API Usage

```python
from chunk import chunk_gated_delta_rule

# Training with chunks
o, final_state = chunk_gated_delta_rule(
    q,      # [B, T, H, K]
    k,      # [B, T, H, K] - typically L2 normalized
    v,      # [B, T, H, V]
    g,      # [B, T, H] - decay gates in log space
    beta,   # [B, T, H] - update gates
    scale=1/sqrt(K),
    initial_state=h0,  # [B, H, K, V]
    output_final_state=True,
)
```

## References

- [Gated Delta Networks Paper (arXiv 2412.06464)](https://arxiv.org/abs/2412.06464)
- [DeltaNet Explained Blog](https://sustcsonglin.github.io/blog/2024/deltanet-1/)
- [GitHub: NVlabs/GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet)
- [Original DeltaNet Paper (NeurIPS 2024)](https://yzhang.site/assets/pubs/neurips/2024/deltanet.pdf)
- Authors: Songlin Yang, Yu Zhang (and collaborators)
