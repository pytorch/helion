# Mamba (Selective State Space Model) Analysis

---
## **MATHEMATICAL CORE**

> **Selective (input-dependent) State Space Model:**
>
> ```
> dt_t = softplus(dt_projection(x_t) + dt_bias)   # Input-dependent discretization
> B_t = B_projection(x_t)                          # Input-dependent input matrix
> C_t = C_projection(x_t)                          # Input-dependent output matrix
>
> dA_t = exp(dt_t * A)                             # Discretized decay
> dB_t = dt_t * B_t                                # Discretized input scaling
>
> h_t = dA_t * h_{t-1} + dB_t * x_t                # State update
> y_t = C_t @ h_t + D * x_t                        # Output
> ```
>
> The key insight: Making `dt`, `B`, `C` **functions of the input** enables content-based reasoning (selection) while preserving linear O(T) complexity. Standard SSMs use fixed parameters, limiting expressivity.

---

## Overview

Mamba is a **selective state space model (SSM)** architecture published by Tri Dao and Albert Gu in December 2023 ([arXiv 2312.00752](https://arxiv.org/abs/2312.00752)). It introduces **data-dependent (selective) gating** to traditional SSMs, enabling content-based reasoning while maintaining linear-time complexity. Mamba achieves **5x higher throughput** than Transformers and matches or exceeds Transformer performance across modalities.

**Mamba-2** (ICML 2024) further optimizes the architecture through **State Space Duality (SSD)**, enabling efficient tensor core utilization by connecting SSMs to linear attention.

## Core Algorithm

### The SSM Foundation

Traditional State Space Models (S4) use fixed, input-independent transition matrices:

```
h_t = A @ h_{t-1} + B @ x_t    # State update
y_t = C @ h_t + D @ x_t         # Output
```

This enables efficient convolution-based computation but lacks content-based reasoning.

### Mamba: Selective SSM

Mamba makes the SSM parameters **input-dependent (selective)**:

```python
def selective_ssm(x, dt, A, B, C, D):
    # dt, B, C are now functions of input x
    for t in range(T):
        dt_t = softplus(dt_projection(x_t) + dt_bias)  # Input-dependent
        B_t = B_projection(x_t)                        # Input-dependent
        C_t = C_projection(x_t)                        # Input-dependent

        dA_t = exp(dt_t * A)                           # Discretization
        dB_t = dt_t * B_t

        h_t = dA_t * h_{t-1} + dB_t * x_t              # State update
        y_t = C_t @ h_t + D * x_t                      # Output
    return y
```

### Mamba-2: State Space Duality (SSD)

Mamba-2 reveals the dual relationship between SSMs and linear attention:

```
SSM: y_t = C_t @ (sum_{s<=t} exp(sum_{s<u<=t} A_u * dt_u) * (B_s * dt_s * x_s))
Linear Attention: y_t = q_t @ (sum_{s<=t} k_s @ v_s)
```

By tying A to be scalar per head, the SSD form enables efficient chunk-wise parallel computation using matrix multiplication.

## Kernel Variants in This Directory

### 1. `ssd_combined.py` - Main Training/Inference Kernel

The primary entry point combining all SSD operations:

```python
def mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=None, z=None, ...):
    """
    Combined chunk-parallel Mamba computation.

    Args:
        x: (batch, seqlen, nheads, headdim) - Input
        dt: (batch, seqlen, nheads) - Discretization timesteps
        A: (nheads,) - Decay rates
        B: (batch, seqlen, ngroups, dstate) - Input matrix
        C: (batch, seqlen, ngroups, dstate) - Output matrix
        D: (nheads, headdim) or (nheads,) - Skip connection
        z: (batch, seqlen, nheads, headdim) - Gate (for SiLU gating)
    """
```

**Forward Pass** (`_mamba_chunk_scan_combined_fwd`):
```python
def _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size, ...):
    # 1. Cumulative sum of log-decay within chunks
    dA_cumsum, dt = _chunk_cumsum_fwd(dt, A, chunk_size, dt_bias, dt_softplus)

    # 2. Compute chunk-level states
    states = _chunk_state_fwd(B, x, dt, dA_cumsum, ...)

    # 3. Pass states between chunks via weighted cumsum
    states, final_states = _state_passing_fwd(states, dA_cumsum[:, :, :, -1], ...)

    # 4. Compute attention-like matrix within chunks
    CB = _bmm_chunk_fwd(C, B, chunk_size, ...)

    # 5. Chunk scan to produce output
    out = _chunk_scan_fwd(CB, x, dt, dA_cumsum, C, states, D, z, ...)

    return out, final_states
```

### 2. `selective_state_update.py` - Single-Step Update

Efficient kernel for autoregressive inference (one token at a time):

```python
@triton.jit
def _selective_scan_update_kernel(
    state_ptr, x_ptr, dt_ptr, A_ptr, B_ptr, C_ptr, D_ptr, z_ptr, out_ptr, ...
):
    # Load current state: [dim, dstate]
    state = tl.load(state_ptrs, ...)

    # Compute discretized parameters
    dt = tl.load(dt_ptrs, ...)
    if DT_SOFTPLUS:
        dt = softplus(dt)
    dA = tl.exp(A * dt)               # Decay
    dB = B * dt                        # Input scaling

    # State update: h_t = dA * h_{t-1} + dB * x_t
    state = state * dA + dB * x[:, None]
    tl.store(state_ptrs, state, ...)

    # Output: y_t = C @ h_t + D * x_t
    out = tl.sum(state * C[None, :], axis=1)
    if HAS_D:
        out += x * D
    if HAS_Z:
        out *= z * tl.sigmoid(z)       # SiLU gating
    tl.store(out_ptrs, out, ...)
```

### 3. `ssd_chunk_state.py` - Chunk State Computation

Computes the state at the end of each chunk:

```python
@triton.jit
def _chunk_state_fwd_kernel(x, b, states, dt, dA_cumsum, ...):
    # For each chunk, compute:
    # state[chunk] = sum_{t in chunk} exp(dA_cumsum[chunk, -1] - dA_cumsum[chunk, t]) * B_t * x_t

    # Load cumsum for weighting
    b_dA_cs = tl.load(dA_cumsum_ptrs, ...)
    dA_cs_last = b_dA_cs[-1]

    # Accumulate weighted B * x
    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
        b_x = tl.load(x_ptrs + k * stride, ...)
        b_b = tl.load(b_ptrs + k * stride, ...)
        b_dt = tl.load(dt_ptrs + k, ...)
        b_dA_cs_k = tl.load(dA_cumsum_ptr + k, ...)

        # exp(log_decay_total - log_decay_at_k)
        b_scale = tl.exp(dA_cs_last - b_dA_cs_k) * b_dt

        # Outer product: B @ X^T weighted by scale
        b_x_scaled = b_x * b_scale[None, :]
        b_states += tl.dot(b_b.T, b_x_scaled)
```

### 4. `ssd_chunk_scan.py` - Chunk Scan Output

Computes the output within each chunk using both:
1. **Inter-chunk**: Query against accumulated state from previous chunks
2. **Intra-chunk**: Attention-like computation within the chunk

### 5. `ssd_state_passing.py` - State Propagation

Propagates states between chunks using a weighted cumulative sum.

### 6. `ssd_bmm.py` - Batch Matrix Multiply

Computes the C @ B^T matrix within each chunk for intra-chunk attention.

### 7. `layernorm_gated.py` - Fused RMSNorm with Gating

Fused kernel for the output normalization with gating:
```python
# RMSNorm(x) * SiLU(z) or RMSNorm(x * SiLU(z))
```

### 8. `k_activations.py` - Activation Kernels

Fused SwiGLU activation kernels for the Mamba block.

## Key Optimizations

### 1. Chunk-Parallel Computation
Instead of sequential recurrence, process chunks of tokens in parallel:
```
Chunk 1: [t0...t63]  →  state_1
Chunk 2: [t64..t127] →  state_2 = f(state_1, chunk_2)
...
```

### 2. Cumulative Sum for Decay
Precompute cumulative log-decay for efficient weighting:
```python
dA_cumsum = cumsum(dt * A)  # Log space
scale = exp(dA_cumsum[-1] - dA_cumsum[k])  # Relative decay
```

### 3. TensorCore Utilization (Mamba-2)
By tying A to be scalar per head, matrix multiplications become the dominant operation, enabling TensorCore acceleration.

### 4. Fused Operations
Multiple operations are fused into single kernels:
- dt_bias + softplus + discretization
- State update + output computation + gating
- RMSNorm + gating

## Comparison: Mamba-1 vs Mamba-2

| Feature | Mamba-1 | Mamba-2 (SSD) |
|---------|---------|---------------|
| A matrix | Per-dimension | Scalar per head |
| Computation | Parallel scan | Matrix multiply |
| TensorCores | Not used | Fully utilized |
| State expansion | N=16 typical | N=64-256 possible |
| FLOP/s | 19 TFLOPS (FP32) | 312 TFLOPS (BF16) |

## Performance Characteristics

### Complexity
- **Time**: O(T * N * D) where N = state dimension, D = head dimension
- **Memory**: O(N * D) per head for hidden state (constant regardless of sequence length)
- **Chunk Size**: Typically 256 tokens

### Speed vs. Attention
```
FlashAttention-2: O(T^2 * D / M) where M = SRAM size
Mamba SSM Scan:   O(T * N * D)

Crossover point: Mamba faster beyond ~2K sequence length
```

## Use Cases

1. **Long-context language modeling**: Linear complexity enables 1M+ contexts
2. **Streaming inference**: Fixed memory state for continuous generation
3. **Audio/Speech**: Natural fit for sequential waveform processing
4. **Hybrid architectures**: Combine with attention layers (e.g., Jamba, Hymba)

## API Usage

```python
from ssd_combined import mamba_chunk_scan_combined

# Training
out = mamba_chunk_scan_combined(
    x,          # [batch, seqlen, nheads, headdim]
    dt,         # [batch, seqlen, nheads]
    A,          # [nheads]
    B,          # [batch, seqlen, ngroups, dstate]
    C,          # [batch, seqlen, ngroups, dstate]
    chunk_size=256,
    D=D,        # [nheads] or [nheads, headdim]
    z=z,        # [batch, seqlen, nheads, headdim] for gating
    dt_bias=dt_bias,
    dt_softplus=True,
    return_final_states=True,
)

# Inference (single step)
from selective_state_update import selective_state_update

out = selective_state_update(
    state,      # [batch, nheads, dim, dstate]
    x,          # [batch, nheads, dim]
    dt,         # [batch, nheads, dim]
    A, B, C, D, z,
    dt_softplus=True,
)
```

## References

- [Mamba Paper (arXiv 2312.00752)](https://arxiv.org/abs/2312.00752)
- [Mamba-2 / SSD Paper (ICML 2024)](https://arxiv.org/abs/2405.21060)
- [State Space Duality Blog - Tri Dao](https://tridao.me/blog/2024/mamba2-part3-algorithm/)
- [GitHub: state-spaces/mamba](https://github.com/state-spaces/mamba)
- Authors: Albert Gu, Tri Dao
