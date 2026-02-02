# Multi-Head Latent Attention (MLA) Analysis

---
## **MATHEMATICAL CORE**

> **Low-rank compression of KV cache with decoupled positional embeddings:**
>
> ```
> c^{KV} = W^{DKV} @ x           # Compress to low-dim latent (d_c << d_h × H)
> K = W^{UK} @ c^{KV}            # Decompress keys on-demand
> V = W^{UV} @ c^{KV}            # Decompress values on-demand
> ```
>
> For positional encoding (Decoupled RoPE):
> ```
> K_h = [W^{UK}_h @ c^{KV} ; RoPE(W^{KR} @ x)]    # Content + position parts
> Score = K_content^T @ Q_content + K_RoPE^T @ Q_RoPE
> ```
>
> **Cache only:** `c^{KV}` (d_c dims) + `k^R` (d_R dims) instead of full K, V (d_h × H dims each).
> DeepSeek-V2/V3: d_c=512, d_R=64, d_h×H=16384 → **32x compression**.

---

## Overview

Multi-Head Latent Attention (MLA) is an attention mechanism introduced in DeepSeek-V2 that addresses the KV cache memory bottleneck in large language models through low-rank factorization. Unlike MQA (Multi-Query Attention) and GQA (Grouped-Query Attention) which reduce the number of attention heads, MLA compresses the KV cache into a low-dimensional latent space while preserving the expressive power of distinct K and V heads for each query head.

## The Problem MLA Solves

In standard Multi-Head Attention (MHA), the KV cache grows linearly with:
- Number of layers
- Sequence length
- Number of heads
- Head dimension

For long-context inference, the KV cache becomes a significant memory bottleneck, limiting batch size and maximum sequence length. MLA reduces this by caching a compressed latent vector instead of full K and V matrices.

## Mathematical Formulation

### Standard MHA (Baseline)

In standard multi-head attention:
```
Q = X · W_Q    (query projection)
K = X · W_K    (key projection)
V = X · W_V    (value projection)
Attention = softmax(Q · K^T / √d_h) · V
```

KV cache per token per layer: `2 × d_h × H` elements (where H = number of heads)

### MLA Core Equations

MLA introduces a low-rank compression/decompression scheme:

**1. Compression (during token generation):**
```
c^{KV}_n = W^{DKV} · x_n
```
Where:
- `c^{KV} ∈ ℝ^{d_c}` is the compressed latent vector
- `W^{DKV} ∈ ℝ^{d_c × d}` is the down-projection matrix
- `d_c << d_h × H` (compression dimension much smaller than full KV dimension)

**2. Decompression (during attention computation):**
```
K = W^{UK} · C^{KV}    (uncompress to keys)
V = W^{UV} · C^{KV}    (uncompress to values)
```
Where:
- `W^{UK}, W^{UV} ∈ ℝ^{(d_h × H) × d_c}` are up-projection matrices

**3. Query Compression:**
```
C^Q = W^{DQ} · X       (compress query)
Q = W^{UQ} · C^Q       (decompress query)
```

### Compression Ratio

The compression ratio is: `r = (d_h × H) / d_c`

- DeepSeek-V2/V3 uses: `d_h = 128`, `H = 128`, `d_c = 512`
- This gives compression ratio `r = 32`, yielding ~20× speedup

### KV Cache Reduction

| Mechanism | Cache per token per layer |
|-----------|--------------------------|
| MHA       | 2 × d_h × H elements     |
| MQA       | 2 × d_h elements         |
| GQA       | 2 × d_h × G elements (G groups) |
| MLA       | d_c elements             |

## Decoupled RoPE (Rotary Position Embeddings)

Standard RoPE is incompatible with MLA because position information must be encoded in the compressed representation. DeepSeek solved this with "Decoupled RoPE":

**Per-Head Construction:**
```
K_h = [ W^{UK}_h · C^{KV}  ]    (content part, no RoPE)
      [ RoPE(W^{KR} · X)   ]    (position part, with RoPE)

Q_h = [ W^{UQ}_h · C^Q        ]    (content part)
      [ RoPE(W^{QR}_h · C^Q)  ]    (position part)
```

Where:
- `W^{KR} ∈ ℝ^{d_R × d}` is shared across all heads (cached separately)
- `W^{QR} ∈ ℝ^{d_R × d_c}` is per-head rotation projection
- `d_R` = rotary embedding dimension (e.g., 64)

**Score Computation:**
```
S_h = (K^{content})^T · Q^{content} + (K^{RoPE})^T · Q^{RoPE}
```

The scaling factor becomes `1/√(d_h + d_R)`.

**What gets cached:**
- `c^{KV}` (compressed KV latent): `d_c` elements
- `k^R` (RoPE component): `d_R` elements
- Total: `d_c + d_R` per token (e.g., 512 + 64 = 576)

## Weight Absorption (Inference Optimization)

During inference, multiple matrix multiplications can be "absorbed" into single operations:

**Query-Key Absorption:**
```
W^{KQ} = (W^{UK})^T · W^{UQ}
S = (C^{KV})^T · W^{KQ} · C^Q
```
This eliminates the need to explicitly decompress K during inference.

**Output-Value Absorption:**
```
W^{OV} = W^O · W^{UV}
Y = W^{OV} · C^{KV} · Z
```

## Kernel Implementations in This Repository

### 1. Dense MLA Decoding (`bench_flash_mla.py`)

The main Triton kernel implements split-K attention for variable-length sequences:

**`_mla_attn_kernel`** (lines 136-219):
- Processes query tokens with separate `q_nope` (content) and `q_pe` (RoPE) components
- Uses paged KV cache with configurable block size
- Implements online softmax for numerical stability
- Supports split-K parallelism across sequence length

Key parameters:
- `BLOCK_H`: Number of heads processed per thread block (default: 16)
- `BLOCK_N`: Number of KV tokens processed per iteration (default: 64)
- `NUM_KV_SPLITS`: Number of splits for parallel reduction
- `HEAD_DIM_CKV`: Compressed KV dimension (512)
- `HEAD_DIM_KPE`: RoPE dimension (64)

**`_mla_softmax_reducev_kernel`** (lines 274-320):
- Combines partial results from split-K computation
- Uses log-sum-exp trick for numerically stable reduction

### 2. Sparse MLA Decoding (`test_flash_mla_sparse_decoding.py`)

Sparse attention variant for efficient long-context inference:
- Supports top-k attention selection
- Includes "attention sink" feature for first few tokens
- Variable-length sequences with per-request configuration

Key features:
- `topk`: Number of attended tokens (e.g., 64-2048)
- `extra_topk`: Secondary top-k for two-stage selection
- `have_topk_length`: Per-request top-k lengths
- `enable_attn_sink`: Keep first tokens always in attention

### 3. Sparse MLA Prefill (`test_flash_mla_sparse_prefill.py`)

Prefill kernel for initial context processing:
- Optimized for compute-bound scenarios
- Supports both d_qk=512 and d_qk=576 configurations

## FlashMLA Library (External CUDA Kernels)

The tests use DeepSeek's FlashMLA library which provides optimized CUDA kernels:

**Supported Operations:**
- Dense decoding (SM90, BF16 KV cache)
- Sparse decoding (SM90/SM100, FP8 KV cache)
- Sparse prefill (SM90/SM100)
- Dense MHA prefill (SM100)

**Performance (H800 SXM5):**
- Dense MLA decoding: up to 3000 GB/s (memory-bound), 660 TFlops (compute-bound)
- Sparse MLA decoding: 410 TFlops
- Sparse prefill: 640 TFlops

**FP8 KV Cache Memory Layout (656 bytes/token):**
- 512 bytes: Quantized NoPE (float8_e4m3)
- 16 bytes: Scale factors (4 × float32 for groups of 128)
- 128 bytes: RoPE part (64 × bfloat16, unquantized)

## Comparison with Other Attention Mechanisms

| Mechanism | KV Cache Size | Distinct Heads | Inference Speed |
|-----------|---------------|----------------|-----------------|
| MHA       | 8 KB/token/layer | Yes | Baseline |
| MQA       | ~0.5 KB/token/layer | No (shared) | Fast |
| GQA       | Variable | Grouped | Medium |
| MLA       | ~0.6 KB/token/layer | Yes (decompressed) | Fast |

MLA achieves MQA-level cache efficiency while maintaining the expressive power of distinct heads per query through on-demand decompression.

## Key Hyperparameters

| Parameter | DeepSeek-V2/V3 Value | Description |
|-----------|---------------------|-------------|
| d_c (r_kv) | 512 | Compressed latent dimension |
| d_R (d_qk_rope) | 64 | RoPE dimension |
| d_h | 128 | Per-head dimension |
| H (h_q) | 128 | Number of query heads |
| h_kv | 1 | Number of KV heads (effectively 1 due to compression) |
| block_size | 64 | Paged KV cache block size |

## References

- [DeepSeek-V2 Paper (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434)
- [Understanding Multi-Head Latent Attention](https://planetbanatt.net/articles/mla.html)
- [DeepSeek's Multi-Head Latent Attention - Lior Sinai](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html)
- [FlashMLA GitHub Repository](https://github.com/deepseek-ai/FlashMLA)
- [DeepSeek-V3 Explained: Multi-head Latent Attention](https://medium.com/data-science/deepseek-v3-explained-1-multi-head-latent-attention-ed6bee2a67c4)
