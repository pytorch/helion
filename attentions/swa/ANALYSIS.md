# Sliding Window Attention (SWA) - Algorithm Analysis

---
## **MATHEMATICAL CORE**

> **Banded attention mask restricting attention to local window:**
>
> ```
> M[i,j] = 0 if |i - j| ≤ W/2, else -∞
> O = softmax((Q @ K^T / √d) + M) @ V
> ```
>
> Token at position `i` attends only to positions in `[i - W/2, i + W/2]`.
>
> **Complexity reduction:**
> - Standard attention: O(N² × d) time, O(N²) memory
> - Sliding window: O(N × W × d) time, O(N × W) memory
>
> **Trade-off:** Information from distant tokens propagates through L/W layers. Mitigated by attention sinks (Mistral), global tokens (Longformer), or interleaved full/SWA layers.

---

## Overview

Sliding Window Attention (SWA) is a **sparse attention mechanism** that restricts each token to attend only to a local window of neighboring tokens, rather than the full sequence. This reduces the time complexity from O(n²) to O(w × n), where w is the window size and n is the sequence length.

## Origins and Key Papers

SWA was popularized by several influential works:
- **Longformer** (Beltagy et al., 2020) - First introduced systematic sliding window attention for long documents
- **Sparse Transformers** (Child et al., 2019) - Explored sparse attention patterns including local windows
- **Mistral 7B** (2023) - Combined SWA with "attention sinks" for efficient inference in LLMs

## Core Algorithm

### Attention Mask Pattern

For a sequence of length `n` and window size `w`, the sliding window creates a **banded attention mask**:

```
Position:  0  1  2  3  4  5  6  7  (seq_len=8, window=4, half_window=2)
    0     [1  1  1  0  0  0  0  0]
    1     [1  1  1  1  0  0  0  0]
    2     [1  1  1  1  1  0  0  0]
    3     [0  1  1  1  1  1  0  0]
    4     [0  0  1  1  1  1  1  0]
    5     [0  0  0  1  1  1  1  1]
    6     [0  0  0  0  1  1  1  1]
    7     [0  0  0  0  0  1  1  1]
```

Token at position `i` can attend to positions in range `[i - half_window, i + half_window]`.

### Mathematical Formulation

Standard attention:
```
Attention(Q, K, V) = softmax(QK^T / √d) × V
```

Sliding window attention with mask M:
```
SWA(Q, K, V) = softmax((QK^T / √d) + M) × V
```

Where M is -∞ outside the window and 0 inside.

## Implementation Variants in This Codebase

The implementation in `attentions/swa/kernels/` provides a **Flash Attention 2-based** Triton implementation supporting three modes:

### 1. Global Attention (MODE=0)
- Full attention without any masking
- All tokens attend to all other tokens
- Standard O(n²) attention pattern

### 2. Causal Attention (MODE=1)
- Lower triangular mask (autoregressive)
- Token i can only attend to tokens j where j ≤ i
- Two-stage processing:
  - **Stage 1**: Blocks fully to the left of diagonal (full attention within block)
  - **Stage 2**: Diagonal blocks (transition between masked/unmasked)

### 3. Sliding Window Attention (MODE=2)
- Banded attention pattern around the diagonal
- Three-stage processing:
  - **Stage 1**: Blocks fully inside the window (full attention)
  - **Stage 2**: Right boundary blocks (upper triangle masked)
  - **Stage 3**: Left boundary blocks (lower triangle masked)

## Triton Kernel Architecture

### Forward Pass (`forward.py`)

The forward kernel `_attn_fwd` implements Flash Attention 2 with sliding window support:

**Key optimizations:**
1. **Block-wise tiling**: Processes Q in `BLOCK_SIZE_Q` chunks and K/V in `BLOCK_SIZE_KV` chunks
2. **Online softmax**: Uses the log-sum-exp trick with running max `m_i` and sum `l_i`
3. **Pipelining optimization**: Separates stages to optimize Triton's instruction scheduling

**Block range computation for sliding window:**
```python
half_window = WINDOW_SIZE // 2
window_block_index = ceil((1 + half_window) / BLOCK_SIZE_Q)

# Stage 1: Full attention blocks between diagonals
lower = max(0, block_index_q - window_block_index + 2) * BLOCK_SIZE_Q
higher = min(NUM_BLOCKS_Q, block_index_q + window_block_index - 1) * BLOCK_SIZE_Q
```

**Inner loop (`_attn_fwd_inner`):**
```python
for start_kv in range(lower, higher, BLOCK_SIZE_KV):
    S_block = dot(Q_block, K_T_block)  # Attention scores

    # Apply sliding window mask for boundary blocks
    if STAGE == 2:  # Right diagonal
        mask = q_offsets[:, None] + half_window >= start_kv + kv_offsets[None, :]
    elif STAGE == 3:  # Left diagonal
        mask = q_offsets[:, None] - half_window <= start_kv + kv_offsets[None, :]

    S_block = S_block * softmax_factor + where(mask, 0, -1e6)

    # Online softmax update
    m_ij = maximum(m_i, max(S_block, axis=1))
    P_block = exp(S_block - m_ij[:, None])
    cor_fact = exp(m_i - m_ij)
    l_i = cor_fact * l_i + sum(P_block, axis=1)
    O_block = O_block * cor_fact[:, None] + dot(P_block, V_block)
```

### Backward Pass (`backward.py`)

Three kernels for the backward pass:

1. **`_attn_bwd_preprocess`**: Computes D = rowsum(dO ⊙ O), used for gradient computation

2. **`_attn_bwd_dk_dv`**: Computes gradients for K and V
   - Fixes K, V blocks and iterates through Q blocks
   - For sliding window: masks P_T_block before gradient accumulation

3. **`_attn_bwd_dq`**: Computes gradient for Q
   - Fixes Q blocks and iterates through K, V blocks
   - Applies same window masking as forward pass

**Gradient equations:**
```
dP = dO × V^T
dS = P ⊙ (dP - D)  # where D_i = Σ_j dO_ij × O_ij
dQ = dS × K × softmax_factor
dK = dS^T × Q × softmax_factor
dV = P^T × dO
```

## Complexity Analysis

| Aspect | Standard Attention | Sliding Window |
|--------|-------------------|----------------|
| Time | O(n² × d) | O(n × w × d) |
| Memory | O(n²) | O(n × w) |
| HBM Accesses | O(n × d + n²) | O(n × d + n × w) |

Where:
- n = sequence length
- d = head dimension
- w = window size

## Trade-offs and Considerations

### Advantages
1. **Linear complexity**: O(n × w) instead of O(n²)
2. **Memory efficient**: No full n×n attention matrix needed
3. **Hardware-friendly**: Block structure maps well to GPU SRAM

### Limitations
1. **Limited receptive field**: Each token only sees w neighbors
2. **No global context**: Distant tokens cannot directly interact
3. **Information propagation**: Requires L/w layers for full sequence visibility

### Mitigation Strategies (from literature)
- **Attention sinks**: Preserve initial tokens (Mistral approach)
- **Global tokens**: Add special tokens that attend everywhere (Longformer)
- **Interleaved FA/SWA layers**: Mix full and sliding window layers (SWAA)

## Usage Example

```python
from flash_attention import FlashAttention

# Shape: (batch_size, num_heads, seq_len, head_dim)
Q = torch.randn(2, 8, 1024, 64, device='cuda', dtype=torch.float16)
K = torch.randn(2, 8, 1024, 64, device='cuda', dtype=torch.float16)
V = torch.randn(2, 8, 1024, 64, device='cuda', dtype=torch.float16)

# Global attention
output = FlashAttention.apply(Q, K, V, None, "global")

# Causal attention (autoregressive)
output = FlashAttention.apply(Q, K, V, None, "causal")

# Sliding window attention (window_size=256)
output = FlashAttention.apply(Q, K, V, 256, "sliding_window")
```

## Paged Attention with Sliding Window

Paged attention is a memory management technique for KV caches during LLM inference. It allocates KV cache in fixed-size pages, enabling efficient memory sharing and on-demand allocation. Combined with sliding window attention, paged SWA provides both memory efficiency and computational sparsity.

### Key Optimization: Tile-Level Window Pruning

The paged SWA kernels implement **tile-level pruning** that skips entire KV blocks outside the sliding window:

```python
# For decode, query is at position (seq_len - 1)
# Valid KV positions are in [seq_len - W, seq_len - 1]
window_start = seq_len - sliding_window

# Skip entire splits that are outside the window
if split_start >= split_end after window clamping:
    # Write zeros and -inf, skip processing
    return
```

This provides O(W) complexity instead of O(N), as we avoid computing attention scores for positions outside the window.

### Two Paging Indexing Schemes

The paged SWA kernels support both indexing schemes used in production systems:

#### 1. CSR-Style Paging (from SGLang)

Uses sparse matrix CSR format for compact representation:
```python
# kv_indptr: [batch_size + 1] - cumulative token counts
# kv_indices: [total_tokens] - physical page indices
start = kv_indptr[batch_idx]
end = kv_indptr[batch_idx + 1]
kv_loc = kv_indices[start:end]
```

**Pros:** Compact, no wasted space for short sequences
**Cons:** Indirect indexing overhead

#### 2. Block Table Paging (from vLLM)

Uses a 2D block table for O(1) lookup:
```python
# block_table: [num_seqs, max_num_blocks_per_seq]
physical_block = block_table[seq_idx, logical_block_idx]
token_offset = token_idx % BLOCK_SIZE
```

**Pros:** O(1) random access, simpler indexing
**Cons:** Wastes memory for short sequences

### Paged SWA Kernels

#### Decode Attention (`paged_swa_decode.py`)

Two-stage flash decoding with sliding window:

**Stage 1:** Parallel partial attention over KV splits
- Each split processes a subset of the KV cache
- Applies window mask and tile-level pruning
- Outputs normalized partial attention and log-sum-exp (LSE)

**Stage 2:** Log-sum-exp reduction across splits
- Combines partial results using numerically stable reduction
- Handles -inf LSE values (from pruned splits) gracefully
- Produces final normalized output

```python
# Stage 1 stores:
#   - acc_i: normalized partial attention for split i
#   - lse_i: log(Z_i) where Z_i = sum_t(exp(qk_t))

# Stage 2 combines:
#   acc_combined = (acc_0 * Z_0 + acc_1 * Z_1 + ...) / Z_total
#   = sum_i(acc_i * w_i) where w_i = Z_i / Z_total
```

#### Prefill Attention (`paged_swa_prefill.py`)

Processes multiple query tokens with sliding window:
- Uses query and key start locations for ragged batching
- Applies both causal and sliding window masks
- Tile-level pruning for positions outside window

The extend kernel supports prefix caching:
- Processes new tokens while reusing cached KV
- Handles the boundary between cached and new KV

### File Structure (Updated)

```
attentions/swa/
├── kernels/
│   ├── __init__.py              # Module exports
│   ├── forward.py               # Forward pass kernel (_attn_fwd)
│   ├── backward.py              # Backward pass kernels (_attn_bwd_*)
│   ├── flash_attention.py       # PyTorch autograd wrapper (FlashAttention class)
│   ├── paged_swa_decode.py      # Paged decode with SWA (CSR + block table)
│   └── paged_swa_prefill.py     # Paged prefill/extend with SWA
└── tests/
    ├── test_swa.py              # Tests for non-paged SWA
    ├── test_paged_swa.py        # Tests for paged SWA kernels
    ├── test.py                  # Basic test script
    └── benchmark.py             # Performance benchmarks
```

### Usage Example (Paged SWA)

```python
from swa.kernels import paged_swa_decode_fwd, paged_swa_prefill_fwd

# CSR-style paged decode with sliding window
paged_swa_decode_fwd(
    q,              # [batch, heads, head_dim]
    k_buffer,       # [total_kv_tokens, kv_heads, head_dim]
    v_buffer,       # [total_kv_tokens, kv_heads, head_dim]
    output,         # [batch, heads, head_dim]
    kv_indptr,      # [batch + 1] - CSR row pointers
    kv_indices,     # [total_kv_tokens] - physical indices
    attn_logits,    # [batch, heads, max_splits, head_dim] - temp storage
    attn_lse,       # [batch, heads, max_splits] - temp LSE storage
    num_kv_splits,  # [batch] - splits per sequence
    max_kv_splits,  # Maximum splits
    sm_scale,       # 1/sqrt(head_dim)
    sliding_window, # Window size (e.g., 4096)
)
```

## Autotuning Configuration

The kernels use Triton's autotuning with the following search space:

```python
@triton.autotune([
    triton.Config(
        {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
        num_stages=num_stages,
        num_warps=num_warps,
    )
    for BLOCK_SIZE_Q in [16]
    for BLOCK_SIZE_KV in [16]
    for num_stages in [3, 4, 7]
    for num_warps in [2, 4]
], key=["SEQ_LEN", "HEAD_DIM"])
```

## References

- [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) - Beltagy et al., 2020
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135) - Dao et al., 2022
- [FlashAttention-2: Faster Attention with Better Parallelism](https://arxiv.org/abs/2307.08691) - Dao, 2023
- [Mistral 7B](https://arxiv.org/abs/2310.06825) - Jiang et al., 2023
- [SWAT: Sliding Window Attention Training](https://arxiv.org/abs/2502.18845) - 2025
- [SWAA: Sliding Window Attention Adaptation](https://arxiv.org/abs/2512.10411) - 2025
