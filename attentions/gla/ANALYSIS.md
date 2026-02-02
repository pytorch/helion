# Gated Linear Attention (GLA) Analysis

---
## **MATHEMATICAL CORE**

> **Linear attention with per-key-channel decay gates:**
>
> ```
> h_t = diag(exp(g_t)) @ h_{t-1} + k_t ⊗ v_t
> o_t = scale * (q_t @ h_t)
> ```
>
> Key insight: Unlike standard linear attention `h_t = h_{t-1} + k_t ⊗ v_t` which can only accumulate, GLA's **channel-wise gates** `g_t ∈ ℝ^K` allow selective decay per dimension:
> ```
> h_t[c] = exp(g_t[c]) * h_{t-1}[c] + k_t[c] * v_t    # Per-channel forgetting
> ```
>
> This provides fine-grained control over which features to retain vs. forget.

---

## Overview

Gated Linear Attention (GLA) is a **hardware-efficient linear attention** mechanism published in December 2023 ([arXiv 2312.06635](https://arxiv.org/abs/2312.06635)). It introduces data-dependent gating to vanilla linear attention, enabling per-key-channel forget gates while maintaining O(N) complexity. The FlashLinearAttention implementation outperforms FlashAttention-2 on longer sequences.

## Core Algorithm

### The Linear Attention Foundation

Vanilla linear attention computes:
```
h_t = h_{t-1} + k_t ⊗ v_t
o_t = q_t @ h_t
```

This suffers from information overload - the model can only add associations, never forget.

### GLA: Adding Channel-Wise Gating

GLA adds **per-key-channel** decay gates:
```python
# Recurrent form (naive implementation)
for i in range(T):
    gk_i = gk[:, :, i].exp()           # [B, H, K] - per-channel decay
    h = h * gk_i[..., None]             # Apply decay to [B, H, K, V] state
    h = h + k_i[..., None] * v_i[..., None, :]  # Add new association
    o[:, :, i] = (q_i[..., None] * h).sum(-2)   # Query readout
```

Key difference from Gated DeltaNet: GLA uses **per-key-channel gates** `[B, T, H, K]` while Gated DeltaNet uses **scalar gates** `[B, T, H]` per head.

### Mathematical Formulation

For timestep t:
```
h_t = diag(exp(g_t)) @ h_{t-1} + k_t ⊗ v_t
o_t = scale * (q_t @ h_t)
```

Where:
- `g_t ∈ [B, H, K]` - channel-wise log-space gates
- `h_t ∈ [B, H, K, V]` - hidden state matrix
- `scale = 1/sqrt(K)`

## Kernel Variants in This Directory

### 1. `naive.py` - Reference Implementation

Simple recurrent implementation for correctness verification:

```python
def naive_recurrent_gla(q, k, v, gk, initial_state, output_final_state):
    h = initial_state or zeros(B, H, K, V)
    for i in range(T):
        q_i = q[:, :, i] * scale
        gk_i = gk[:, :, i].exp()         # Per-channel decay
        kv_i = k_i[..., None] * v_i[..., None, :]  # Outer product
        h = h * gk_i[..., None] + kv_i   # Gated update
        o[:, :, i] = (q_i[..., None] * h).sum(-2)  # Query
    return o, h
```

**Complexity**: O(T * K * V) per sequence, fully sequential.

### 2. `chunk.py` - Chunk-Parallel Training

The main training kernel using **chunk-level parallelism**:

**Forward Pass** (`chunk_gla_fwd`):
```python
def chunk_gla_fwd(q, k, v, g, g_cumsum, scale, initial_state, ...):
    # 1. Cumulative sum of gates within chunks
    if g_cumsum is None:
        g_cumsum = chunk_local_cumsum(g, chunk_size)

    # 2. Compute inter-chunk hidden states
    h, ht = chunk_fwd_h(k, v, gk=g_cumsum, h0=initial_state, ...)

    # 3. Compute intra-chunk attention matrix A
    A = chunk_gla_fwd_intra_gk(q, k, g=g_cumsum, scale=scale, ...)

    # 4. Compute output
    o = chunk_gla_fwd_o_gk(q, v, g=g_cumsum, A=A, h=h, scale=scale, ...)

    return g_cumsum, A, h, ht, o
```

**Key Triton Kernels**:

#### `chunk_gla_fwd_A_kernel_intra_sub_inter`
Computes attention scores between sub-blocks within a chunk:
```python
@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_inter(...):
    # Compute gated attention: qg @ kg^T
    b_qg = b_q * exp(b_g - b_gn[None, :]) * scale  # Gate-adjusted query
    b_kg = b_k * exp(b_gn[:, None] - b_gk)          # Gate-adjusted key
    b_A = tl.dot(b_qg, b_kg)                        # Attention scores
```

#### `chunk_gla_fwd_kernel_o`
Computes final output by combining inter-chunk and intra-chunk contributions:
```python
@triton.jit
def chunk_gla_fwd_kernel_o(...):
    # Inter-chunk: query @ hidden_state
    b_qg = (b_q * exp(b_g)).to(b_q.dtype)
    b_o = tl.dot(b_qg, b_h)
    b_o *= scale

    # Intra-chunk: attention @ values
    b_A = tl.where(causal_mask, b_A, 0.)
    b_o += tl.dot(b_A, b_v)
```

**Grid**: `(NV, NT, B*H)` - parallelizes over value blocks, time chunks, and batch*heads.

### 3. `fused_recurrent.py` - Inference Wrapper

Wraps the common fused recurrent kernel for inference:
```python
def fused_recurrent_gla(q, k, v, gk, gv, scale, initial_state, ...):
    return fused_recurrent(
        q=q, k=k, v=v, g=None, gk=gk, gv=gv, ...
    )
```

Supports both key-wise (`gk`) and value-wise (`gv`) gating.

### 4. `fused_chunk.py` - Fused Chunk Kernel

Optimized fused version combining multiple operations for better memory efficiency.

## Comparison with Gated DeltaNet

| Feature | GLA | Gated DeltaNet |
|---------|-----|----------------|
| Gate dimensions | `[B, T, H, K]` (per-channel) | `[B, T, H]` (per-head) |
| Update rule | Additive | Delta (error-correcting) |
| Hidden state | `h = g*h + kv` | `h = g*h + k⊗(β*(v - h@k))` |
| Parameters | More (K gates per head) | Fewer (1 gate per head) |
| Expressiveness | Higher channel control | Better memory modification |

## Performance Characteristics

### Complexity
- **Time**: O(T * K * V) total, O(T/C) chunks parallelizable
- **Memory**: O(K * V) per head for hidden state
- **Chunk Size**: Typically 64 tokens

### Key Optimizations
1. **Chunk-level cumsum**: Precompute `g_cumsum = cumsum(g)` within chunks
2. **Split-merge for large K**: When K > 256, split computation across K blocks
3. **TF32 precision**: Uses TF32 for inter-sub-block attention
4. **Exp2 optimization**: Optional use of `exp2` instead of `exp` for speed

## Use Cases

1. **Long-context language modeling**: GLA excels at length generalization (2K→20K+)
2. **Efficient inference**: Recurrent form enables streaming with fixed memory
3. **Alternative to Mamba**: Competitive with Mamba while being purely attention-based
4. **Hybrid architectures**: Can interleave with standard attention layers

## API Usage

```python
from chunk import chunk_gla

# Training with chunks
o, final_state = chunk_gla(
    q,      # [B, T, H, K]
    k,      # [B, T, H, K]
    v,      # [B, T, H, V]
    g,      # [B, T, H, K] - per-channel log-space gates
    scale=1/sqrt(K),
    initial_state=h0,  # [B, H, K, V]
    output_final_state=True,
)
```

## References

- [GLA Paper (arXiv 2312.06635)](https://arxiv.org/abs/2312.06635)
- [Flash Linear Attention GitHub](https://github.com/fla-org/flash-linear-attention)
- [FlashAttention-style algorithm](https://arxiv.org/abs/2312.06635)
- Authors: Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, Yoon Kim
