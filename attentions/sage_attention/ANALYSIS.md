# SageAttention Analysis

---
## **MATHEMATICAL CORE**

> **INT8 quantized QK matmul with K smoothing for accuracy preservation:**
>
> ```
> K_smooth = K - mean(K, dim=seq)             # Remove channel-wise outliers
> Q̂, K̂ = quantize_int8(Q / √d), quantize_int8(K_smooth)
> S = dequantize(Q̂ @ K̂^T)                    # INT8 matmul + scale
> O = softmax(S) @ V                          # FP16 PV matmul
> ```
>
> **Why K smoothing is exact:**
> ```
> softmax(Q @ (K - μ)^T) = softmax(Q @ K^T - Q @ μ^T) = softmax(Q @ K^T)
> ```
> Subtracting a constant from all attention scores doesn't change softmax output.
>
> Result: **2-5x speedup** over FlashAttention by using INT8 Tensor Cores (4x faster than FP16 on consumer GPUs).

---

## Overview

SageAttention is a quantized attention mechanism that achieves 2-5x speedup over FlashAttention by leveraging INT8 matrix multiplication for the QK computation while maintaining end-to-end model accuracy. The key insight is that INT8 Tensor Core operations on consumer GPUs (e.g., RTX 3090, 4090) are 4x faster than FP16 and 2x faster than FP8.

**Papers:**
- SageAttention (INT8): [arXiv:2410.02367](https://arxiv.org/abs/2410.02367)
- SageAttention2 (INT4): [arXiv:2411.10958](https://arxiv.org/abs/2411.10958)
- Accepted at ICLR 2025, ICML 2025, and NeurIPS 2025 Spotlight

## Algorithm

### Standard Attention

Standard attention computes:
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d)) * V
```

where Q, K, V are query, key, and value matrices of shape `[batch, heads, seq_len, head_dim]`.

### SageAttention Formulation

SageAttention quantizes the attention computation as:

1. **Quantization Step:**
   ```
   (δQ, Q̂) = ψQ(Q / sqrt(d))   # Quantize scaled queries to INT8
   (δK, K̂) = φK(K)              # Quantize keys to INT8 (with smoothing)
   ```

2. **INT8 Attention Score Computation:**
   ```
   S = ψ⁻¹(Q̂ @ K̂^T)            # INT8 matmul, then dequantize
     = (δQ * δK) * (Q̂ @ K̂^T)   # Scale factors applied after matmul
   ```

3. **Softmax and Output:**
   ```
   P = softmax(S)
   O = P @ V                    # FP16 matmul with FP16/FP32 accumulator
   ```

### The Problem: Channel-wise Outliers in K

Simply quantizing Q and K to INT8 causes severe accuracy degradation:
- Text-to-image models (e.g., UniDiffuser) produce completely blurry images
- LLMs (e.g., Llama2) achieve only random-guessing accuracy (25.5% on MMLU)

**Root Cause:** The K matrix exhibits significant channel-wise outliers that lead to substantial quantization error.

### Solution: K Smoothing

SageAttention proposes a **K smoothing** technique:

```
K_smooth = K - mean(K)
```

where `mean(K) = (1/N) * Σ K[t,:]` is the mean across the sequence dimension.

**Mathematical Justification:** This transformation preserves attention outputs because:
```
softmax(Q @ (K - mean(K))^T) = softmax(Q @ K^T - Q @ mean(K)^T)
                              = softmax(Q @ K^T)  # constant shift doesn't affect softmax
```

The overhead is less than 0.2% of total computation time.

## Kernel Implementations in This Repository

### 1. Per-Block INT8 Quantization (`quant_per_block.py`)

Quantizes Q and K at block granularity (BLOCK_M=128 for Q, BLOCK_N=64 for K):

```python
# For each block of tokens:
scale = max(|x|) / 127
x_int8 = round(x * sm_scale / scale)
```

**Key aspects:**
- One scale factor per block of tokens
- Query scaling includes `sm_scale = 1/sqrt(d)` multiplied by `1.44269504` (log2(e) for exp2 optimization)
- Symmetric quantization: maps [-127, 127] range

### 2. Per-Thread INT8 Quantization (`quant_per_thread.py`)

Finer-grained quantization aligned with Tensor Core thread layout:

**For Queries:**
- 8 tokens per warp share one scale (8 scales per 128-token block)
- Each thread handles tokens at stride-8 positions

**For Keys:**
- 2 adjacent tokens share one scale (4 scales per 64-token block)
- Optimized for the transposed access pattern in K^T

This per-thread granularity improves OPS by ~12% without sacrificing accuracy.

### 3. Non-Causal Attention Kernel (`attn_qk_int8_per_block.py`)

FlashAttention-style tiled attention with INT8 QK matmul:

```python
@triton.jit
def _attn_fwd_inner(...):
    for start_n in range(0, kv_len, BLOCK_N):
        # Load INT8 Q and K blocks
        k = tl.load(K_ptrs)
        k_scale = tl.load(K_scale_ptr)

        # INT8 matmul + dequantization
        qk = tl.dot(q, k).to(tl.float32) * (q_scale * k_scale)

        # Online softmax
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])  # exp2 for speed

        # FP16 PV matmul with accumulator update
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(tl.float16), v, out_dtype=tl.float16)
```

**Key optimizations:**
- Uses `exp2` instead of `exp` (faster on GPU, scales adjusted accordingly)
- FP16 accumulator for PV matmul (2x faster than FP32, with periodic fp32 buffer flush)
- Supports optional attention masks

### 4. Causal Attention Kernel (`attn_qk_int8_per_block_causal.py`)

Two-stage processing for causal masking:

```python
# Stage 1: Process full blocks (no mask needed)
for start_n in range(0, start_m * BLOCK_M, BLOCK_N):
    # Full blocks - all tokens are valid

# Stage 2: Process diagonal block (apply causal mask)
for start_n in range(start_m * BLOCK_M, (start_m + 1) * BLOCK_M, BLOCK_N):
    mask = offs_m[:, None] >= (start_n + offs_n[None, :])
    qk += tl.where(mask, 0, float('-inf'))
```

## API Variants

The implementation provides multiple API functions optimized for different GPU architectures:

| Function | QK Precision | PV Precision | Accumulator | Target GPU |
|----------|-------------|--------------|-------------|------------|
| `sageattn_qk_int8_pv_fp16_triton` | INT8 | FP16 | FP16+FP32 | SM86 (RTX 3090) |
| `sageattn_qk_int8_pv_fp16_cuda` | INT8 | FP16 | FP16/FP32 | SM80+ |
| `sageattn_qk_int8_pv_fp8_cuda` | INT8 | FP8 | FP32 | SM89 (RTX 4090) |
| `sageattn_qk_int8_pv_fp8_cuda_sm90` | INT8 | FP8 | FP32 | SM90 (H100) |

## Accumulator Strategies

### PV Accumulation Options

1. **FP32 Accumulator:** Most accurate but slower due to CUDA core overhead
2. **FP16 Accumulator:** 2x faster, may have precision issues with large biases
3. **FP16+FP32 Hybrid:** FP16 accumulator flushed to FP32 buffer periodically

### V Smoothing (for FP16 Accumulator)

When using pure FP16 accumulation, V smoothing helps maintain accuracy:
```python
if smooth_v:
    v_smooth = v - mean(v)  # Remove bias
    # Accumulate in FP16
    # Add back bias at the end
```

## Performance Characteristics

| Metric | FlashAttention2 | SageAttention |
|--------|-----------------|---------------|
| RTX 4090 TOPS (headdim=128) | 165 | 340 |
| Speedup | 1x | ~2.7x |
| Peak INT8 Utilization | N/A | 52% |

## Supported Configurations

- **Head dimensions:** 64, 128 (padded if smaller)
- **Tensor layouts:** HND (head-first) or NHD (sequence-first)
- **GQA support:** `num_qo_heads` must be divisible by `num_kv_heads`
- **Data types:** FP16, BF16 input (V converted to FP16 for Triton kernel)

## Key Files

| File | Description |
|------|-------------|
| `core.py` | Main API entry points with automatic GPU dispatch |
| `attn_qk_int8_per_block.py` | Triton kernel for non-causal attention |
| `attn_qk_int8_per_block_causal.py` | Triton kernel for causal attention |
| `quant_per_block.py` | Block-level INT8 quantization kernel |
| `quant_per_thread.py` | Thread-level INT8 quantization kernel (finer granularity) |

## Mathematical Correctness

The key insight is that K smoothing is mathematically exact:

```
softmax(QK^T) = softmax(Q(K - μ)^T + Qμ^T)
              = softmax(Q(K - μ)^T)  # Adding constant to all elements doesn't change softmax
```

When returning log-sum-exp (LSE) for Ring Attention compatibility, the correction term `Qμ^T * sm_scale` is added back.
