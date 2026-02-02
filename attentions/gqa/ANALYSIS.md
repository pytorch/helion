# Grouped Query Attention (GQA) Analysis

---
## **MATHEMATICAL CORE**

> **Sharing KV heads across groups of query heads:**
>
> ```
> kv_group_num = H_q // H_kv
>
> For each query head h:
>     kv_head = h // kv_group_num
>     O[h] = softmax(Q[h] @ K[kv_head]^T / √d) @ V[kv_head]
> ```
>
> The key insight: **Query heads within a group share the same K and V**, reducing KV cache from `2 × H_q × d` to `2 × H_kv × d` per token.
>
> Special cases:
> - GQA-1 (H_kv=1): Multi-Query Attention (MQA)
> - GQA-H (H_kv=H_q): Standard Multi-Head Attention (MHA)

---

## Overview

Grouped Query Attention (GQA) is a **memory-efficient attention mechanism** published by Google in May 2023 ([arXiv 2305.13245](https://arxiv.org/abs/2305.13245)). It interpolates between Multi-Head Attention (MHA) and Multi-Query Attention (MQA) by sharing key-value heads across groups of query heads, reducing KV cache size by up to 90% while maintaining near-MHA quality.

GQA has been widely adopted in production LLMs including **Llama 2/3**, **Mistral 7B**, and **IBM Granite 3.0**.

## Core Algorithm

### The KV Cache Problem

In autoregressive LLM inference:
- Standard MHA stores `[seq_len, num_heads, head_dim]` for both K and V
- With 32 heads and long sequences, this becomes a memory bottleneck
- MQA (1 KV head) is too aggressive, causing quality degradation

### GQA Solution: Grouped KV Heads

GQA divides `H_q` query heads into `G` groups, each sharing one K and one V head:

```
kv_group_num = H_q // H_kv

For each query head h:
    kv_head = h // kv_group_num
    attn = softmax(Q[h] @ K[kv_head]^T) @ V[kv_head]
```

**Key configurations**:
- GQA-1 (G=1, H_kv=1): Equivalent to MQA
- GQA-H (G=H_q, H_kv=H_q): Equivalent to MHA
- GQA-G: Intermediate, e.g., 8 KV heads for 32 query heads (G=4)

## Kernel Variants in This Directory

### 1. `triton_decode_attention.py` - Paged Decode Attention

**Purpose**: Memory-efficient decoding with paged KV cache and GQA support.

This kernel implements **split-KV flash decoding** for autoregressive generation:

#### Stage 1: Split-KV Attention (`_fwd_kernel_stage1`)

```python
@triton.jit
def _fwd_kernel_stage1(
    Q, K_Buffer, V_Buffer, sm_scale,
    Req_to_tokens, B_Seqlen, Att_Out,
    kv_group_num, BLOCK_N, NUM_KV_SPLITS, PAGE_SIZE, ...
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num  # GQA head mapping

    # Split KV cache across splits for parallelism
    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = min(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    # FlashAttention-style online softmax
    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
        # Paged memory access
        kv_page_number = tl.load(Req_to_tokens + cur_batch * stride + offs_n // PAGE_SIZE)
        kv_loc = kv_page_number * PAGE_SIZE + offs_n % PAGE_SIZE

        # Load K, V from paged buffer
        k = tl.load(K_Buffer + kv_loc * stride + cur_kv_head * ...)
        v = tl.load(V_Buffer + ...)

        # Attention computation with online softmax
        qk = tl.sum(q[None, :] * k, 1) * sm_scale
        if logit_cap > 0:
            qk = logit_cap * tanh(qk / logit_cap)  # Logit capping

        n_e_max = tl.maximum(tl.max(qk, 0), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max)
        acc = acc * re_scale + tl.sum(p[:, None] * v, 0)
        e_sum = e_sum * re_scale + tl.sum(p, 0)
        e_max = n_e_max

    # Store partial result and LSE
    tl.store(Att_Out + offs_mid_o, acc / e_sum)
    tl.store(Att_Out + offs_mid_o_1, e_max + tl.log(e_sum))  # LSE
```

**Grid**: `(batch, head_num, NUM_KV_SPLITS)`

#### Stage 1 Grouped Variant (`_fwd_grouped_kernel_stage1`)

Optimized for GQA with multiple query heads processed together:

```python
@triton.jit
def _fwd_grouped_kernel_stage1(..., BLOCK_H: tl.constexpr):
    # Process BLOCK_H query heads sharing the same KV head
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)

    # Matrix multiply for multiple heads at once
    qk = tl.dot(q, k.to(q.dtype))  # [BLOCK_H, BLOCK_N]
    acc += tl.dot(p.to(v.dtype), v)  # [BLOCK_H, BLOCK_DV]
```

**Key optimization**: Batches multiple query heads that share the same KV head.

#### Stage 2: Reduce Across Splits (`_fwd_kernel_stage2`)

Merges partial results from Stage 1 using LSE-weighted combination:

```python
@triton.jit
def _fwd_kernel_stage2(Mid_O, o, lse, B_Seqlen, NUM_KV_SPLITS, ...):
    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    for split_kv_id in range(NUM_KV_SPLITS):
        tv = tl.load(Mid_O + offs_v + split_kv_id * stride)
        tlogic = tl.load(Mid_O + offs_logic + split_kv_id * stride)  # LSE

        n_e_max = tl.maximum(tlogic, e_max)
        old_scale = tl.exp(e_max - n_e_max)
        exp_logic = tl.exp(tlogic - n_e_max)

        acc = acc * old_scale + exp_logic * tv
        e_sum = e_sum * old_scale + exp_logic
        e_max = n_e_max

    tl.store(o + cur_batch * stride + cur_head * stride + offs_d, acc / e_sum)
```

### 2. `triton_prefill_attention.py` - Prefill/Context Attention

**Purpose**: Standard FlashAttention-style prefill with GQA and sliding window support.

```python
@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale, B_Start_Loc, B_Seqlen, Out,
    kv_group_num, BLOCK_M, BLOCK_N, IS_CAUSAL,
    SLIDING_WINDOW_Q, SLIDING_WINDOW_K, ...
):
    cur_kv_head = cur_head // kv_group_num  # GQA mapping

    # Load query block
    q = tl.load(Q + off_q, mask=..., other=0.0)

    # FlashAttention-style blocked computation
    for start_n in range(start_n_limit, end_n_limit, BLOCK_N):
        # Causal + sliding window mask
        mask = pos_k < cur_batch_seq_len
        if IS_CAUSAL:
            mask &= pos_q >= pos_k
        if SLIDING_WINDOW_Q > 0:
            mask &= pos_q - pos_k <= SLIDING_WINDOW_Q
        if SLIDING_WINDOW_K > 0:
            mask &= pos_k - pos_q <= SLIDING_WINDOW_K

        # Attention computation
        k = tl.load(k_ptrs + start_n * stride_kbs, ...)
        qk = tl.dot(q, k)
        qk = tl.where(mask, qk * sm_scale, -1.0e8)

        # Online softmax update
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_ij
```

**Features**:
- Bidirectional sliding window support (`SLIDING_WINDOW_Q`, `SLIDING_WINDOW_K`)
- Uses `exp2` instead of `exp` for speed (with `RCP_LN2` scaling)
- Supports MLA-style head dimensions (576, 288, 192)

## Memory Layout & Paging

### Paged KV Cache
```python
# Page table: maps logical positions to physical pages
kv_page_number = Req_to_tokens[batch_idx, token_idx // PAGE_SIZE]
physical_loc = kv_page_number * PAGE_SIZE + token_idx % PAGE_SIZE

# K/V buffers are indexed by physical location
K_Buffer[physical_loc, kv_head, head_dim]
```

### Intermediate Storage for Split-KV
```python
# Stage 1 output: partial attention + LSE
Att_Out[batch, head, split_id, head_dim + 1]  # +1 for LSE
```

## Performance Characteristics

| Metric | MHA | GQA-8 | MQA |
|--------|-----|-------|-----|
| KV Heads | 32 | 8 | 1 |
| KV Cache Size | 100% | 25% | 3.1% |
| Inference Speed | 1x | 1.3-1.4x | 1.4-1.5x |
| Quality | Baseline | ~Baseline | -1-2% |

### Split-KV Benefits
- Parallelizes across KV splits for long sequences
- Each split computes partial attention with LSE
- Stage 2 merges with O(num_splits) work

### ROCm/HIP Optimizations
```python
if is_hip_:
    BLOCK = 8  # Smaller blocks for AMD
    extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}
    num_stages = 1
```

## Use Cases

1. **Long-context LLM inference**: Reduced KV cache enables longer contexts
2. **High-throughput serving**: Smaller memory footprint → larger batch sizes
3. **Edge deployment**: Memory-constrained environments
4. **vLLM/SGLang integration**: Paged attention for efficient serving

---

## Paged GQA Attention

This directory also contains **paged attention** variants that combine GQA with paged KV cache management. Paging is a memory management strategy (like OS virtual memory) that reduces KV cache waste from 60-80% to <4%.

### Why Paged KV Cache?

In LLM serving, sequences have variable lengths and the KV cache grows during generation:
- **Without paging**: Pre-allocate max_seq_len per request → massive memory waste
- **With paging**: Allocate fixed-size blocks on-demand → near-zero waste

### Two Indexing Schemes

We provide two paging implementations, each from a major serving framework:

| Scheme | Structure | Trade-off | Origin |
|--------|-----------|-----------|--------|
| **Block Table** | 2D `[num_seqs, max_blocks]` | O(1) lookup, but wastes space for short seqs | vLLM |
| **CSR** | `indptr[batch+1]` + `indices[total]` | Compact, but requires indirection | SGLang |

### Paged Kernels - CSR Style (from SGLang)

CSR indexing uses two arrays:
```python
kv_indptr: [batch_size + 1]  # Cumulative token counts (prefix sum)
kv_indices: [total_tokens]   # Physical page indices

# To access KV for sequence i, token j:
start = kv_indptr[i]
page_idx = kv_indices[start + j]
k = k_buffer[page_idx]
```

**Kernel files:**
- `paged_decode_csr.py`: Two-stage flash decoding with split-KV parallelism
- `paged_prefill_csr.py`: Varlen prefill with b_start_loc/b_seq_len batching
- `extend_attention_csr.py`: Prefix caching (prefix + extend regions)

**Flash Decoding Algorithm (CSR):**
```python
# Stage 1: Parallel partial attention over KV splits
grid = (batch, num_heads, MAX_KV_SPLITS)
for each split:
    partial_output, lse = compute_attention(kv_slice)
    store(partial_output, lse)

# Stage 2: LSE-weighted reduction
grid = (batch, num_heads)
for each split:
    acc = acc * exp(e_max - n_e_max) + exp(split_lse - n_e_max) * split_output
output = acc / e_sum
```

### Paged Kernels - Block Table Style (from vLLM)

Block table uses 2D array for O(1) lookup:
```python
block_table: [num_seqs, max_num_blocks_per_seq]

# To access KV for sequence i, token t:
logical_block = t // block_size
physical_block = block_table[i, logical_block]
offset = t % block_size
k = k_cache[physical_block, offset, ...]
```

**Kernel files:**
- `paged_decode_blocktable.py`: Unified 2D/3D attention with parallel segments
- `paged_prefill_blocktable.py`: Chunked prefill + paged decode

**KV Cache Layouts (Block Table):**
```python
# 5D K layout for coalesced memory access
K_cache: [num_blocks, num_kv_heads, head_size//x, block_size, x]  # x=8 typically

# 4D V layout
V_cache: [num_blocks, num_kv_heads, head_size, block_size]
```

The 5D K layout groups adjacent dimensions (`head_size//x` and `x`) to enable coalesced reads when multiple threads access consecutive dimensions.

**3D Kernel with Parallel Segments:**

For very long sequences, the 3D kernel splits KV into parallel segments:
```python
# Grid: (num_q_blocks, num_kv_heads, num_segments)
kernel_unified_attention_3d:
    segm_idx = tl.program_id(2)
    tiles_per_segment = cdiv(seq_len, num_segments * TILE_SIZE)

    # Each segment computes partial attention
    for j in range(segm_idx * tiles_per_segment, (segm_idx + 1) * tiles_per_segment):
        # ... attention computation

    # Store partial output + LSE for reduction
    tl.store(segm_output_ptr, acc)
    tl.store(segm_max_ptr, M)
    tl.store(segm_expsum_ptr, L)

# Reduction kernel combines segments
reduce_segments:
    overall_max = max(segm_max)
    segm_expsum = segm_expsum * exp(segm_max - overall_max)
    output = sum(segm_output * exp(segm_max - overall_max)) / sum(segm_expsum)
```

### Extend Attention (Prefix Caching)

For chunked prefill and prefix caching scenarios:
```
Prefix: Previously computed KV in paged buffer
Extension: New tokens being processed

Query must attend to both regions.
```

Two-stage computation:
```python
# Stage 1: Query attends to PREFIX (from KV buffer)
for block in prefix:
    k, v = load_from_buffer(kv_indices)
    compute_attention()

# Stage 2: Query attends to EXTENSION (from current batch)
for block in extension:
    k, v = load_from_extend_tensors()
    apply_causal_mask()
    compute_attention()
```

### Sliding Window + Paging

Both paging schemes support sliding window attention with **tile-level pruning**:
```python
# Skip entire tiles outside sliding window - major optimization
first_allowed_key = context_len + qpos_lo - SLIDING_WINDOW + 1
last_allowed_key = context_len + qpos_hi
tile_start = max(0, first_allowed_key // TILE_SIZE)
tile_end = min((last_allowed_key // TILE_SIZE) + 1, num_tiles)

# Only iterate tiles in [tile_start, tile_end)
for j in range(tile_start, tile_end):
    # ... process tile
```

### Additional Features

Both paging implementations support:
- **GQA/MQA**: Head grouping via `kv_group_num`
- **FP8 quantization**: Per-block scales for KV cache
- **ALiBi position encoding**: Linear attention bias
- **Logit capping/Softcap**: For Gemma-style models
- **Multimodal prefix ranges**: Bidirectional attention for image tokens

---

## API Usage

### Non-Paged (Dense) Attention

```python
from attentions.gqa.kernels import context_attention_fwd, triton_decode_attention_fwd

# Prefill attention (variable-length batched)
context_attention_fwd(
    q, k, v, o,
    b_start_loc,    # [batch] start locations
    b_seq_len,      # [batch] sequence lengths
    max_input_len,
    is_causal=True,
)

# Decode attention
triton_decode_attention_fwd(
    q, k_buffer, v_buffer, o, lse,
    req_to_token, b_seq_len, attn_logits,
    num_kv_splits=8, sm_scale=1/sqrt(head_dim),
)
```

### Paged Attention - CSR Style (SGLang)

```python
from attentions.gqa.kernels import (
    paged_decode_csr_fwd,
    paged_prefill_csr_fwd,
    extend_attention_csr_fwd,
)

# Paged decode with CSR indexing
paged_decode_csr_fwd(
    q,              # [batch, num_heads, head_dim]
    k_buffer,       # [total_pages, page_size, num_kv_heads, head_dim]
    v_buffer,       # [total_pages, page_size, num_kv_heads, head_dim]
    o,              # [batch, num_heads, head_dim]
    kv_indptr,      # [batch + 1] cumulative token counts
    kv_indices,     # [total_tokens] physical page indices
    b_seq_len,      # [batch] sequence lengths
    ...
)

# Extend attention (prefix caching)
extend_attention_csr_fwd(
    q, k_extend, v_extend, o,
    k_buffer, v_buffer,
    kv_indptr, kv_indices,
    b_seq_len, b_prefix_len, b_extend_len,
    ...
)
```

### Paged Attention - Block Table Style (vLLM)

```python
from attentions.gqa.kernels import (
    paged_decode_blocktable_fwd,
    chunked_prefill_paged_decode_blocktable,
)

# Unified paged attention with block table
paged_decode_blocktable_fwd(
    q, k_cache, v_cache, out,
    cu_seqlens_q,           # [num_seqs + 1] cumulative query lengths
    max_seqlen_q,
    seqused_k,              # [num_seqs] KV sequence lengths
    max_seqlen_k,
    softmax_scale,
    causal=True,
    window_size=(-1, -1),   # (left, right), -1 = no limit
    block_table,            # [num_seqs, max_num_blocks]
    softcap=0.0,
    ...
)

# Chunked prefill + paged decode (mixed batch)
chunked_prefill_paged_decode_blocktable(
    query, key, value, output,
    kv_cache_dtype,
    key_cache,              # [num_blocks, num_kv_heads, head_size//x, block_size, x]
    value_cache,            # [num_blocks, num_kv_heads, head_size, block_size]
    block_table,            # [num_seqs, max_num_blocks]
    query_start_loc,
    seq_lens,
    max_seq_len,
    max_query_len,
    k_scale, v_scale,
    ...
)
```

## References

### GQA
- [GQA Paper (arXiv 2305.13245)](https://arxiv.org/abs/2305.13245)
- [ACL Anthology](https://aclanthology.org/2023.emnlp-main.298/)
- Authors: Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yinfei Yang, Santiago Ontañón, Sumit Sanghai

### Paged Attention
- [PagedAttention Paper (SOSP 2023)](https://arxiv.org/abs/2309.06180) - vLLM's paged KV cache
- [vLLM Blog: Easy, Fast, and Cheap LLM Serving](https://blog.vllm.ai/2023/06/20/vllm.html)
- [vLLM V1 on AMD GPUs with Triton](https://pytorch.org/blog/enabling-vllm-v1-on-amd-gpus-with-triton/)
- [Anatomy of vLLM](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html)

### Flash Attention & Flash Decoding
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [Flash-Decoding for long-context inference](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)
- [Online Softmax Notes (UW CSE599M)](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)

### SGLang
- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [LightLLM](https://github.com/ModelTC/lightllm) - Original source for SGLang's Triton kernels
- [Towards Deterministic Inference in SGLang](https://lmsys.org/blog/2025-09-22-sglang-deterministic/)

### vLLM
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [NVIDIA TensorRT-LLM Chunked Prefill](https://developer.nvidia.com/blog/streamlining-ai-inference-performance-and-deployment-with-nvidia-tensorrt-llm-chunked-prefill/)
