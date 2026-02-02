# RWKV Attention Analysis

---
## **MATHEMATICAL CORE**

> **RWKV-6: Exponentially-decayed linear attention with bonus term:**
>
> ```
> h_t = h_{t-1} * exp(w_t) + k_t ⊗ v_t       # Decay + outer product update
> o_t = sum((h_t + u * k_t ⊗ v_t) * r_t)     # Readout with current-token bonus
> ```
>
> **RWKV-7: SGD-inspired dynamic state evolution (NC1 expressivity):**
>
> ```
> S_t = diag(exp(-exp(w_t))) @ S_{t-1}       # Diagonal decay
>     + S_{t-1} @ α_t @ β_t^T                 # Matrix transformation (from gradient ∂L/∂S)
>     + k_t ⊗ v_t                             # Standard KV update
> o_t = S_t @ r_t
> ```
>
> The `S @ α β^T` term simulates online gradient descent: L = ½||v - k^T S||², enabling **in-context learning**.

---

RWKV (Receptance Weighted Key Value) is a novel architecture that combines the benefits of RNNs and Transformers. It achieves Transformer-level performance with RNN-style O(1) inference complexity, enabling efficient processing of arbitrarily long sequences with constant memory.

## Algorithm Overview

RWKV leverages a linear attention mechanism that can be formulated as either:
- **Transformer mode**: Parallelizable computation during training
- **RNN mode**: Sequential computation with constant memory during inference

The key insight is that exponentials enable dual formulations: `exp(w * k)` in parallel mode becomes multiplicative decay `exp(-tw)` in RNN mode.

### Core Components

| Component | Symbol | Description |
|-----------|--------|-------------|
| Receptance | `r` (or `q`) | Query vector - determines what information to retrieve from memory |
| Key | `k` | Importance weight for storing information (exponentiated) |
| Value | `v` | The actual information to store |
| Decay | `w` | Time-dependent decay rate (in log space) |
| Bonus | `u` | Special weight for current token contribution |

## RWKV-6 "Finch" Architecture

RWKV-6 introduced data-dependent decay rates, making the architecture more expressive.

### Mathematical Formulation

**State Update (Recurrent Form):**
```
h_t = h_{t-1} * exp(w_t) + k_t ⊗ v_t
o_t = sum((h_t + u * k_t ⊗ v_t) * r_t)
```

Where:
- `h_t ∈ R^{K×V}` is the hidden state (KV cache analog)
- `w_t ∈ R^K` is the data-dependent decay (in log space)
- `k_t ⊗ v_t` is the outer product creating a rank-1 update
- `u` is the "bonus" parameter for current-token emphasis

**Naive Recurrent Implementation** (`recurrent_naive.py:28-36`):
```python
for i in range(T):
    q_i = q[:, :, i, :] * scale
    k_i = k[:, :, i]
    v_i = v[:, :, i, :]
    w_i = w[:, :, i].exp()  # Convert from log space
    kv_i = k_i[..., None] * v_i[..., None, :]  # Outer product
    o_i = (h + u[None, ..., None] * kv_i) * q_i[..., None]
    o[:, :, i] = o_i.sum(-2)
    h = h * w_i[..., None] + kv_i  # Decay + update
```

### Kernel Variants

#### 1. Fused Recurrent Kernel (`fused_recurrent.py`)

Optimized Triton kernel for element-wise recurrence. Uses tiling to efficiently compute the recurrence across the K and V dimensions.

**Key optimizations:**
- Block-based processing with `BK`, `BV` tiles
- Autotuning for num_warps (1, 2, 4, 8, 16)
- Support for variable-length sequences via `cu_seqlens`
- Separate forward/backward kernels with checkpointing

**Core loop** (`fused_recurrent.py:84-91`):
```python
for _ in range(0, T):
    b_kv = b_k[:, None] * b_v[None, :]
    b_o = tl.sum((b_h + b_kv * b_u[:, None]) * b_q[:, None], 0)
    b_h = b_h * exp(b_w)[:, None] + b_kv
```

#### 2. Chunked Parallel Kernel (`chunk.py`)

Hybrid approach combining parallel intra-chunk attention with sequential inter-chunk state passing.

**Algorithm:**
1. Divide sequence into chunks of size `BT` (typically 32-64)
2. Compute cumulative decay within each chunk
3. **Intra-chunk**: Parallel attention with decay-weighted queries and keys
4. **Inter-chunk**: Sequential hidden state propagation

**Cumulative Decay** (`chunk.py:93-119`):
```python
# Compute inclusive and exclusive cumsum of decay
gi, ge = chunk_rwkv6_fwd_cumsum(g, chunk_size, cu_seqlens)
```

**Inter-chunk attention** uses the GLA (Gated Linear Attention) infrastructure for efficient hidden state computation.

#### 3. Naive Chunk Implementation (`chunk_naive.py`)

Reference implementation showing the chunked algorithm without optimization:

```python
# Inter-chunk: propagate state between chunks
for i in range(num_chunk - 1):
    wkv_new[:, :, i+1] = (wkv_new[:, :, i] * w_cumsum[:, :, i, -1].exp()) + wkv[:, :, i]

# Intra-chunk: causal attention within chunk
for i in range(chunk_size):
    attn = (q[:, :, :, i] * k * exp(w_cumsum[:, :, :, i] - w - w_cumsum)).sum(-1)
    mask.masked_fill_(~(arange < i), 0)  # Causal mask
```

## RWKV-7 "Goose" Architecture

RWKV-7 introduces **Dynamic State Evolution** through a fundamentally new mechanism inspired by online gradient descent.

### Mathematical Foundation

RWKV-7 simulates gradient descent on an internal model `v ≈ k^T S`:

**L2 Loss:**
```
L = (1/2) ||v - k^T S||^2
```

**Gradient:**
```
∂L/∂S = S k k^T - v k^T
```

**State Update (SGD-inspired):**
```
S_t = S_{t-1} D_t + S_{t-1} α_t β_t^T + v_t k_t^T
```

Where:
- `D_t = diag(exp(-exp(w_t)))` - Diagonal decay matrix
- `α_t β_t^T` - Generalized rank-1 state transformation (from `-η k k^T` term)
- `v_t k_t^T` - Standard value-key outer product update

**Output:**
```
o_t = S_t r_t
```

This formulation enables RWKV-7 to achieve **NC1 expressivity**, surpassing the TC0 limitations of standard attention mechanisms.

### Key Innovations

1. **Generalized Delta Rule**: The state transformation `S @ α β^T` allows learning complex temporal patterns
2. **In-context Learning**: Simulates gradient descent during inference
3. **Dynamic Learning Rate**: The `a` and `b` (α, β) parameters modulate how state evolves

### Kernel Variants

#### 1. Fused Recurrent RWKV-7 (`rwkv7/fused_recurrent.py`)

Forward kernel implementing the generalized state evolution:

**Core computation** (`fused_recurrent.py:98-100`):
```python
b_act_a = -b_kk  # α = -k̂ (normalized k)
b_b = b_kk * b_a  # β = k̂ * a

# State update: S = decay * S + (S @ α) * β^T + k ⊗ v
b_h = exp(b_w)[:, None] * b_h + b_b[:, None] * tl.sum(b_act_a[:, None] * b_h, 0)[None, :]
b_h += b_k[:, None] * b_v[None, :]
b_o = tl.sum(b_h * b_r[:, None], 0)
```

The term `(S @ α) * β^T` implements the matrix transformation by:
1. Computing `sa = S @ α` (matrix-vector product)
2. Computing `sa * β^T` (outer product reconstruction)

#### 2. Channel Mixing (`rwkv7/channel_mixing.py`)

RWKV's feed-forward network replacement using:
- Token shift mixing: `k = x + (x_prev - x) * x_k`
- Squared ReLU activation: `relu(x)^2`

**Triton kernel for mixing** (`channel_mixing.py:47-79`):
```python
# Token shift with learnable mixing weights
prev_value = where(is_first, prev_state, prev_x)
state_diff = prev_value - curr_x
mixed = state_diff * k_value
result = curr_x + mixed
```

#### 3. Auxiliary Kernels

- **`fused_addcmul.py`**: Fused multiply-add operations
- **`fused_k_update.py`**: Key vector updates
- **`gate_output_correction.py`**: Output gating corrections

## Comparison: RWKV-6 vs RWKV-7

| Feature | RWKV-6 | RWKV-7 |
|---------|--------|--------|
| State Update | `S = decay * S + k ⊗ v` | `S = decay * S + S @ α β^T + k ⊗ v` |
| Expressivity | Linear attention (TC0) | Dynamic state evolution (NC1) |
| Bonus Term | Yes (`u` parameter) | Incorporated into α, β |
| Decay | Data-dependent per-channel | Data-dependent per-channel |
| In-context Learning | Limited | Explicit SGD simulation |

## Performance Characteristics

### Memory Complexity

| Mode | Memory |
|------|--------|
| Training (parallel) | O(B * T * H * (K + V)) |
| Inference (recurrent) | O(B * H * K * V) - constant per sequence |

### Computational Complexity

| Mode | Complexity |
|------|------------|
| Training (chunked) | O(B * T * H * K * V / chunk_size) |
| Inference (recurrent) | O(B * T * H * K * V) per step |

## Implementation Notes

### Variable-Length Sequence Support

All kernels support variable-length batching via `cu_seqlens` (cumulative sequence lengths), compatible with FlashAttention-style APIs:

```python
# For batch with 4 sequences of varying lengths
cu_seqlens = tensor([0, 100, 350, 600, 1024])
# Sequence 0: tokens 0-99, Sequence 1: tokens 100-349, etc.
```

### Numerical Stability

- Decay `w` is stored in log-log space: `actual_decay = exp(-exp(w))`
- State values kept in FP32 for accumulation
- Chunked kernels maintain cumulative decay in FP32

### Backward Pass Considerations

- RWKV-6 backward requires separate kernels for dq, dk, dv, dw, du
- Gradient of decay w.r.t. w: `dw = -sum(dS * S_prev) * exp(w) * exp(-exp(w))`
- State checkpointing every 16-32 tokens for memory efficiency

## References

- [RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048)
- [RWKV Architecture Wiki](https://wiki.rwkv.com/basic/architecture.html)
- [Full Stack Deep Learning - RWKV Explainer](https://fullstackdeeplearning.com/blog/posts/rwkv-explainer/)
- [RWKV-LM GitHub Repository](https://github.com/BlinkDL/RWKV-LM)
- [Flash Linear Attention (FLA) Library](https://github.com/fla-org/flash-linear-attention)
