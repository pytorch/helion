# Native Sparse Attention (NSA) - Algorithm Analysis

---
## **MATHEMATICAL CORE**

> **Three-branch gated sparse attention:**
>
> ```
> o_t = g_cmp * Attn(q_t, K_cmp, V_cmp)   # Compressed (coarse global)
>     + g_slc * Attn(q_t, K_slc, V_slc)   # Selected (fine-grained important)
>     + g_swa * Attn(q_t, K_swa, V_swa)   # Sliding window (local)
> ```
>
> Key components:
> - **Compression**: `K_cmp = mean_pool(K, block_size)` - coarse-grained global context
> - **Selection**: `top_k_blocks = topk(softmax(q @ K_cmp^T))` - identifies important token blocks
> - **Sliding window**: Local context with window size W
> - **Learned gates** `g_*`: Dynamic weighting of each branch
>
> Complexity: O(T × (T/BS + S×BS + W) × d) where S<<T/BS, achieving ~30x speedup at 64K context.

---

## Overview

Native Sparse Attention (NSA) is a hardware-aligned and natively trainable sparse attention mechanism introduced by DeepSeek in February 2025. NSA addresses the quadratic complexity problem in standard attention by combining three complementary attention strategies through a learned gating mechanism.

**Key Innovation**: NSA achieves efficient long-context modeling (64k+ tokens) while maintaining model performance through a dynamic hierarchical sparse strategy that balances global context awareness with local precision.

**Paper**: [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/abs/2502.11089)

---

## Core Algorithm

### The Three-Branch Architecture

NSA processes input sequences through three parallel attention branches, each capturing different aspects of the context:

```
o_t = g_cmp * Attn(q_t, K_cmp, V_cmp)   # Compressed (coarse-grained global)
    + g_slc * Attn(q_t, K_slc, V_slc)   # Selected (fine-grained important)
    + g_swa * Attn(q_t, K_swa, V_swa)   # Sliding window (local context)
```

Where:
- `g_cmp`, `g_slc`, `g_swa` are learned gate scores in [0, 1]
- Each branch uses standard scaled dot-product attention

---

## Branch 1: Compressed Attention (Coarse-Grained Global)

### Purpose
Captures global context patterns by compressing sequential token blocks into summary representations.

### Algorithm

**Compression Function** (Mean Pooling):
```python
def compression(k, v, block_size):
    # Compress keys and values via mean pooling over blocks
    num_blocks = ceil(T / block_size)
    k_cmp = k.view(B, num_blocks, block_size, H, D).mean(dim=2)  # [B, C, H, D]
    v_cmp = v.view(B, num_blocks, block_size, H, D).mean(dim=2)  # [B, C, H, D]
    return k_cmp, v_cmp
```

**Attention Computation**:
```python
# For query at position t:
# Only attend to completed compression blocks (causal)
NC = (t + 1) // block_size  # Number of complete blocks before t

attn_cmp = softmax(q_t @ K_cmp[:NC].T / sqrt(d))  # [NC]
o_cmp = attn_cmp @ V_cmp[:NC]  # [D]
```

### Mathematical Formulation
```
K_cmp = {mean(k_{i*BS:(i+1)*BS}) | 0 <= i < ceil(T/BS)}
V_cmp = {mean(v_{i*BS:(i+1)*BS}) | 0 <= i < ceil(T/BS)}

For token t:
  NC_t = floor((t+1) / BS)  # completed blocks
  o_cmp_t = g_cmp_t * softmax(q_t @ K_cmp[:NC_t]) @ V_cmp[:NC_t]
```

### Kernel Implementation
- **File**: `parallel.py:parallel_nsa_compression_fwd_kernel`
- Uses block pointers for efficient memory access
- Implements online softmax for numerical stability
- Supports variable-length sequences via `cu_seqlens`

---

## Branch 2: Selected Attention (Fine-Grained Important Tokens)

### Purpose
Attends to the most relevant token blocks identified through importance scoring derived from compression attention.

### Algorithm

**Block Selection via Top-K**:
```python
def select_important_blocks(q, k_cmp, lse_cmp, block_counts, block_size):
    # Compute importance scores from compression attention
    p_cmp = exp(q @ k_cmp.T - lse_cmp)  # [C] attention weights

    # Select top-n blocks based on importance
    # Local block (containing current token) gets max score
    importance = p_cmp.masked_fill(local_mask, 1.0)
    block_indices = importance.topk(block_counts)[1]  # [S]

    return block_indices
```

**Selected Attention Computation**:
```python
def selected_attention(q, k, v, block_indices, block_size):
    o_slc = 0
    for block_idx in block_indices:
        if block_idx * block_size <= current_pos:
            start = block_idx * block_size
            end = min(start + block_size, current_pos + 1)
            k_block = k[start:end]
            v_block = v[start:end]
            attn = softmax(q @ k_block.T / sqrt(d))
            o_slc += attn @ v_block
    return o_slc
```

### Top-K Selection with Bitonic Sort

The implementation uses a Triton-based bitonic sort for efficient top-k selection:

```python
# From utils.py: _bitonic_merge
# Implements parallel bitonic sorting network
# - O(log^2(n)) parallel comparisons
# - Fully parallelizable on GPU
```

**Bitonic Sort Properties**:
- Compare-and-swap network with predictable memory access patterns
- Well-suited for GPU parallelism
- Used to efficiently select top-S blocks from C candidates

### Kernel Implementation
- **File**: `parallel.py:parallel_nsa_kernel_topk` - Block selection
- **File**: `parallel.py:parallel_nsa_fwd_kernel` - Selected attention forward
- Uses bitonic merge sort for top-k selection
- Iterates over selected blocks with causal masking

---

## Branch 3: Sliding Window Attention (Local Context)

### Purpose
Preserves fine-grained local context within a sliding window around each token.

### Algorithm
```python
def sliding_window_attention(q_t, k, v, window_size, pos):
    start = max(0, pos - window_size + 1)
    k_window = k[start:pos+1]
    v_window = v[start:pos+1]

    attn = softmax(q_t @ k_window.T / sqrt(d))
    o_swa = attn @ v_window
    return o_swa
```

### Implementation
- Leverages FlashAttention's sliding window mode for efficiency
- **File**: `parallel.py:parallel_nsa` lines 1423-1440
- Uses `flash_attn_func` or `flash_attn_varlen_func` with `window_size` parameter

---

## Gating Mechanism

### Purpose
Learns to dynamically weight the contributions of each branch based on input context.

### Implementation
```python
# Gates are input-dependent scalars per head
# Typically computed as: g = sigmoid(MLP(input))

# In kernel:
o = o_slc * g_slc.unsqueeze(-1)  # [B, T, HQ, D]
o = torch.addcmul(o, o_cmp, g_cmp.unsqueeze(-1))
o = torch.addcmul(o, o_swa, g_swa.unsqueeze(-1))
```

**Shape Analysis**:
- `g_cmp, g_slc, g_swa`: `[B, T, HQ]` - one gate per query head per token
- Gates enable the model to adaptively balance:
  - Global vs local context
  - Coarse vs fine-grained attention
  - Different patterns for different query positions

---

## Kernel Variants in This Implementation

### 1. Naive Implementation (`naive.py`)

**Functions**:
- `naive_nsa`: Reference implementation for selected + sliding window attention
- `naive_nsa_compression`: Compression attention with block selection
- `naive_nsa_with_compression`: Full NSA pipeline

**Characteristics**:
- PyTorch-based, easy to understand
- Token-by-token processing
- Used as reference for correctness testing

### 2. Parallel Triton Implementation (`parallel.py`)

**Forward Kernels**:
| Kernel | Purpose |
|--------|---------|
| `parallel_nsa_compression_fwd_kernel` | Compressed attention forward |
| `parallel_nsa_kernel_topk` | Top-k block selection |
| `parallel_nsa_fwd_kernel` | Selected attention forward |

**Backward Kernels**:
| Kernel | Purpose |
|--------|---------|
| `parallel_nsa_compression_bwd_kernel_dq` | dQ for compression |
| `parallel_nsa_compression_bwd_kernel_dkv` | dK, dV for compression |
| `parallel_nsa_bwd_kernel_dq` | dQ for selected attention |
| `parallel_nsa_bwd_kernel_dkv` | dK, dV for selected attention |

**Utility Kernels**:
| Kernel | Purpose |
|--------|---------|
| `parallel_nsa_kernel_mask` | Generate block mask for backward |
| `parallel_nsa_bwd_kernel_preprocess` | Compute delta for backward |

### 3. Utility Functions (`utils.py`)

- `_compare_and_swap`: Core bitonic sort primitive
- `_bitonic_merge`: Merge step of bitonic sort
- `argsort`: Full bitonic argsort for top-k selection

---

## Hardware Optimizations

### Memory Access Patterns
```python
# Block pointer setup for coalesced access
p_q = tl.make_block_ptr(q + (bos + i_t) * HQ*K,
                        (HQ, K), (K, 1),
                        (i_h * G, 0), (G, BK), (1, 0))
```

### Tiling Strategy
- **BK**: Key/Query dimension tile (64-256 based on GPU capability)
- **BV**: Value dimension tile (64-256)
- **BS**: Block size for compression (typically 32-64)
- **BC**: Chunk size for iteration (same as BS)

### Autotuning
```python
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4]
    ],
    key=['BS', 'BK', 'BV'],
)
```

### GQA Support
- Supports Grouped Query Attention with ratio `G = HQ // H`
- Constraint: `HQ % (H * 16) == 0` (group size must be multiple of 16)
- Keys/values are broadcast across query head groups

---

## Complexity Analysis

### Standard Attention
- Time: O(T^2 * d)
- Memory: O(T^2)

### NSA
- **Compression**: O(T * C * d) where C = T/BS << T
- **Selection**: O(T * S * BS * d) where S << C
- **Sliding Window**: O(T * W * d) where W = window_size

**Total**: O(T * (C + S*BS + W) * d) = O(T * N_t * d) where N_t << T

### Speedup
For 64k context:
- Standard attention: 64k * 64k = 4B operations
- NSA with S=16, BS=64, W=64: 64k * (1k + 1k + 64) ~ 130M operations
- **~30x theoretical speedup**

---

## Variable-Length Sequence Support

### FlashAttention-Compatible API
```python
# cu_seqlens: cumulative sequence lengths [0, len1, len1+len2, ...]
# token_indices: [[seq_id, pos_in_seq], ...] for each token
# chunk_indices: [[seq_id, chunk_id], ...] for each chunk
```

### Implementation
- Kernels check `USE_OFFSETS` heuristic
- Compute `bos` (begin of sequence) and `eos` (end of sequence) dynamically
- Adjust block pointers and loop bounds per-sequence

---

## Usage Example

```python
from attentions.nsa.kernels import parallel_nsa

# Inputs
q = torch.randn(B, T, HQ, D)  # Queries
k = torch.randn(B, T, H, D)   # Keys (GQA: H < HQ)
v = torch.randn(B, T, H, D)   # Values

# Gates (learned, typically from MLP)
g_cmp = torch.sigmoid(gate_proj_cmp(x))  # [B, T, HQ]
g_slc = torch.sigmoid(gate_proj_slc(x))  # [B, T, HQ]
g_swa = torch.sigmoid(gate_proj_swa(x))  # [B, T, HQ]

# Forward pass
o = parallel_nsa(
    q=q, k=k, v=v,
    g_cmp=g_cmp, g_slc=g_slc, g_swa=g_swa,
    block_counts=16,      # Number of blocks to select
    block_size=64,        # Compression/selection block size
    window_size=64,       # Sliding window size
    scale=1/sqrt(D),
)
```

---

## Key Differences from Other Sparse Attention

| Method | Selection | Trainable | Hardware-Aligned |
|--------|-----------|-----------|------------------|
| Longformer | Fixed patterns | No | Partial |
| BigBird | Fixed + random | No | No |
| Sparse Transformer | Fixed strided | No | No |
| **NSA** | Learned dynamic | Yes | Yes |

**NSA Advantages**:
1. **Native trainability**: End-to-end differentiable through all components
2. **Dynamic selection**: Adapts to input content
3. **Hardware alignment**: Designed for GPU memory hierarchy
4. **GQA support**: Works with modern LLM architectures

---

## References

1. Yuan, J., et al. (2025). "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention." arXiv:2502.11089
2. DeepSeek-AI announcement: https://x.com/deepseek_ai/status/1891745487071609327
3. FLA (Flash Linear Attention) library: Original source of kernel implementations
