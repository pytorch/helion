# Kimi Delta Attention (KDA) Analysis

---
## **MATHEMATICAL CORE**

> **Channel-wise gating + delta rule - combining the best of GLA and Gated DeltaNet:**
>
> ```
> S_t = diag(exp(g_t)) @ S_{t-1} + k_t ⊗ (β_t * (v_t - (k_t @ S_{t-1})))
> o_t = q_t @ S_t
> ```
>
> Key innovations:
> - **Per-channel gates** `g_t ∈ ℝ^K` (from GLA): Fine-grained decay control
> - **Delta rule** `v_t - (k_t @ S_{t-1})` (from DeltaNet): Error-correcting updates
> - **Learnable gate parameters** A_log, dt_bias with numerical stability bounds
>
> Gate computation with safe bounds: `gate = lower_bound * sigmoid(exp(A_log) * g)` constrains values to [-5, 0) for TensorCore acceleration.

---

## Overview

Kimi Delta Attention (KDA) is a **linear attention mechanism with channel-wise gating and delta rule** published by Moonshot AI in their "Kimi Linear" technical report (October 2025). It combines the best of Gated Linear Attention (GLA) and Gated DeltaNet, achieving up to **6x higher decoding throughput** at 1M context length while maintaining quality competitive with full attention.

KDA is the core component of **Kimi Linear**, which interleaves KDA with periodic full attention layers in a 3:1 ratio.

## Core Algorithm

### Key Innovations

1. **Channel-wise gating**: Per-key-channel decay gates `[B, T, H, K]` (like GLA)
2. **Delta rule**: Error-correcting memory updates (like Gated DeltaNet)
3. **Hardware-efficient gating**: Learnable A_log and dt_bias for numerical stability

### Mathematical Formulation

The recurrent form:
```python
def naive_recurrent_kda(q, k, v, g, beta, scale, initial_state):
    S = initial_state or zeros(B, H, K, V)  # Hidden state
    for i in range(T):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]

        # 1. Apply channel-wise decay
        S = S * g_i[..., None].exp()

        # 2. Delta rule update with beta gate
        error = v_i - (k_i[..., None] * S).sum(-2)  # Retrieval error
        S = S + einsum('bhk, bhv -> bhkv', b_i * k_i, error)

        # 3. Query readout
        o[:, i] = einsum('bhk, bhkv -> bhv', q_i, S)
    return o, S
```

### Gate Computation

KDA uses a learnable gate with numerical stability bounds:

```python
def kda_gate(g, A_log, dt_bias, lower_bound=None):
    # Standard form:
    gate = -exp(A_log) * softplus(g + dt_bias)

    # Safe gate form (when lower_bound is set):
    gate = lower_bound * sigmoid(exp(A_log) * g)  # e.g., lower_bound=-5
```

The safe gate constrains values to `[-5, 0)` for TensorCore M=16 acceleration.

## Kernel Variants in This Directory

### 1. `naive.py` - Reference Implementations

Two reference implementations for correctness verification:

#### Recurrent Form
```python
def naive_recurrent_kda(q, k, v, g, beta, ...):
    for i in range(T):
        S = S * g_i[..., None].exp()
        S = S + einsum('bhk, bhv -> bhkv',
                       b_i * k_i,
                       v_i - (k_i[..., None] * S).sum(-2))
        o[:, i] = einsum('bhk, bhkv -> bhv', q_i, S)
```

#### Chunk Form (for training)
```python
def naive_chunk_kda(q, k, v, g, beta, chunk_size=64):
    g = g.cumsum(-2)  # Cumsum within chunks

    # Compute WY-like matrix A for delta rule encoding
    for i in range(BT):
        A[..., i] = einsum('...cd, ...d -> ...c', k * (g - g_i).exp(), k_i)
    A = solve_lower_triangular(-A * beta + I)

    # Transform v through A
    w = A @ (g.exp() * k)
    u = A @ v

    # Inter-chunk propagation
    for chunk_i in range(NT):
        v_i = u_i - w_i @ S
        o[chunk_i] = (q_i * g_i.exp()) @ S + A_qk @ v_i
        S = S * g_i[-1].exp() + k_i^T @ v_i
```

### 2. `chunk.py` - Chunk-Parallel Training Kernel

Main training implementation using chunk-level parallelism:

```python
def chunk_kda_fwd(q, k, v, g, beta, scale, initial_state, ...):
    # 1. Compute intra-chunk matrices (Aqk for Q@K, Akk for delta rule)
    w, u, qg, kg, Aqk, Akk = chunk_kda_fwd_intra(
        q, k, v, gk=g, beta=beta, scale=scale, ...
    )

    # 2. Inter-chunk state propagation (reuses GLA's chunk_h kernel)
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=kg, w=w, u=u, gk=g, initial_state=initial_state, ...
    )

    # 3. Compute output (reuses GLA's output kernel)
    o = chunk_gla_fwd_o_gk(
        q=q, v=v_new, g=g, A=Aqk, h=h, scale=scale, ...
    )

    return o, Aqk, Akk, final_state, ...
```

**Key difference from GLA**: KDA computes two attention matrices:
- `Aqk`: Q @ K^T for output computation
- `Akk`: K @ K^T for delta rule transformation

### 3. `gate.py` - Fused Gate Kernels

Efficient Triton kernels for gate computation with autograd support:

```python
@triton.jit
def kda_gate_fwd_kernel(g, A_log, dt_bias, yg, lower_bound, ...):
    b_A = tl.load(A_log + i_h)
    b_g = tl.load(p_g, ...)

    if HAS_BIAS:
        b_g = b_g + tl.load(dt_bias + ...)

    if not USE_LOWER_BOUND:
        b_yg = -exp(b_A) * softplus(b_g)  # Standard gate
    else:
        b_yg = lower_bound * tl.sigmoid(exp(b_A) * b_g)  # Safe gate

    tl.store(p_yg, b_yg)
```

#### Fused Gate + Cumsum Kernel
```python
@triton.jit
def kda_gate_chunk_cumsum_vector_kernel(...):
    # Compute gate
    if not USE_LOWER_BOUND:
        b_gate = -exp(b_A) * softplus(b_s)
    else:
        b_gate = lower_bound * tl.sigmoid(exp(b_A) * b_s)

    # Apply chunk-local cumsum
    b_o = tl.cumsum(b_gate, axis=0)

    if HAS_SCALE:
        b_o *= scale  # RCP_LN2 for exp2 conversion
```

### 4. `chunk_intra.py` - Intra-Chunk Kernels

Computes the WY-like transformation matrices within chunks.

### 5. `chunk_bwd.py` - Backward Kernels

Gradient computation kernels including:
- `chunk_kda_bwd_dAv`: Gradient w.r.t. attention and values
- `chunk_kda_bwd_wy_dqkg_fused`: Fused gradient computation

### 6. `wy_fast.py` - WY Transform

Efficient WY representation computation for the delta rule encoding.

### 7. `fused_recurrent.py` - Inference Kernel

Wrapper for fused recurrent computation during inference.

## Comparison with Related Methods

| Feature | GLA | Gated DeltaNet | KDA |
|---------|-----|----------------|-----|
| Gate dimensions | `[B,T,H,K]` | `[B,T,H]` | `[B,T,H,K]` |
| Update rule | Additive | Delta | Delta |
| Learnable gate params | No | No | Yes (A_log, dt_bias) |
| Safe gate option | No | No | Yes (lower_bound) |

## Performance Characteristics

### Complexity
- **Time**: O(T * K * V) total, O(T/C) parallel chunks
- **Memory**: O(K * V) per head hidden state (fixed regardless of seq length)
- **Chunk Size**: 64 tokens

### Key Optimizations
1. **exp2 with RCP_LN2**: Uses `exp2(x * RCP_LN2)` instead of `exp(x)` for speed
2. **Safe gate mode**: Enables M=16 TensorCore when gate values are in [-5, 0)
3. **Recomputation vs caching**: Trades compute for memory with `disable_recompute` flag
4. **Variable-length support**: cu_seqlens + chunk_indices for packed sequences

## Use Cases

1. **Ultra-long context**: 1M+ context with constant memory per token
2. **Hybrid architectures**: 3:1 ratio of KDA to full attention in Kimi Linear
3. **Multimodal integration**: Handles mixed-modality sequences efficiently
4. **Streaming inference**: Fixed-memory state enables continuous generation

## API Usage

```python
from chunk import chunk_kda

# Training with optional gate computation in kernel
o, final_state = chunk_kda(
    q,              # [B, T, H, K]
    k,              # [B, T, H, K] - typically L2 normalized
    v,              # [B, T, H, V]
    g,              # [B, T, H, K] - raw gate input or pre-computed
    beta,           # [B, T, H] - update gate
    scale=1/sqrt(K),
    initial_state=h0,   # [B, H, K, V]
    output_final_state=True,
    use_qk_l2norm_in_kernel=True,   # Fuse L2 norm
    use_gate_in_kernel=True,         # Fuse gate computation
    A_log=A_log,                     # [H] learnable
    dt_bias=dt_bias,                 # [H*K] optional bias
    safe_gate=True,
    lower_bound=-5.0,
)

# Inference with intermediate states
with torch.inference_mode():
    o, final_state, h = chunk_kda(
        ...,
        return_intermediate_states=True,  # Returns chunk-level h for caching
    )
```

## References

- [Kimi Linear Technical Report (arXiv 2510.26692)](https://arxiv.org/pdf/2510.26692)
- [Flash Linear Attention (fla) Repository](https://github.com/fla-org/flash-linear-attention)
- [Gated DeltaNet Paper (ICLR 2025)](https://arxiv.org/abs/2412.06464)
- [GLA Paper](https://arxiv.org/abs/2312.06635)
- Authors: Moonshot AI Team, Songlin Yang, Yu Zhang
