# Quantization Kernel Analysis

## Overview

Quantization kernels enable efficient inference by reducing precision of weights and activations. This reduces memory bandwidth requirements and enables faster computation through integer/low-precision arithmetic.

## Mathematical Foundations

### Symmetric Quantization

For symmetric quantization with scale factor $s$:
$$x_q = \text{round}\left(\frac{x}{s}\right)$$
$$\tilde{x} = x_q \cdot s$$

Scale factor is typically computed as:
$$s = \frac{\max(|x|)}{2^{b-1} - 1}$$

where $b$ is the bit width.

### Asymmetric Quantization

For asymmetric quantization with scale $s$ and zero-point $z$:
$$x_q = \text{round}\left(\frac{x}{s}\right) + z$$
$$\tilde{x} = (x_q - z) \cdot s$$

### Per-Token Quantization

Each row (token) has its own scale:
$$x_{q,i,:} = \text{round}\left(\frac{x_{i,:}}{s_i}\right)$$

where $s_i = \frac{\max(|x_{i,:}|)}{127}$ for INT8.

### Per-Group Quantization

Divide features into groups of size $g$ with separate scales:
$$x_{q,i,j} = \text{round}\left(\frac{x_{i,j}}{s_{i,\lfloor j/g \rfloor}}\right)$$

### Block-wise Quantized Matmul (W8A8)

For quantized matmul $C = A \times B$ with block scales:
$$C_{ij} = \sum_k A_{q,ik} \cdot B_{q,kj} \cdot s_A^{(k)} \cdot s_B^{(k)}$$

## Quantization Formats

### INT8
- Range: [-128, 127]
- Symmetric quantization: max_val = 127
- Good balance of accuracy and speed

### FP8 (E4M3/E5M2)
- E4M3: 4 exponent bits, 3 mantissa bits (higher precision)
- E5M2: 5 exponent bits, 2 mantissa bits (larger range)
- Native hardware support on H100+

### INT4 (AWQ/GPTQ)
- 4-bit integer, packed 8 per int32
- Weight-only quantization
- Requires dequantization before compute

## Kernel Implementations

### int8_kernel_sglang.py
- **Functions**: `per_token_quant_int8`, `per_token_group_quant_int8`, `w8a8_block_int8_matmul`
- **Features**:
  - Dynamic per-token quantization
  - Block-wise matmul with scales
  - Optional row-sum computation for bias correction

### fp8_kernel_sglang.py
- **Functions**: `per_token_group_quant_fp8`, `static_quant_fp8`, `w8a8_block_fp8_matmul`
- **Features**:
  - FP8 E4M3/E5M2 format support
  - Static and dynamic quantization
  - MxFP8 microscaling format

### awq_triton_vllm.py
- **Functions**: `awq_dequantize_triton`, `awq_gemm_triton`
- **Features**:
  - 4-bit packed weight dequantization
  - Group-wise scale and zero-point
  - Fused dequant + GEMM

### triton_scaled_mm_vllm.py
- **Functions**: `triton_scaled_mm`
- **Features**:
  - Generic scaled matmul
  - Scale swizzling for tensor cores
  - Multi-precision output support

## Optimization Strategies

1. **Fused Dequant+Compute**: Avoid materializing full-precision weights
2. **Block Alignment**: Align quantization blocks to GEMM tiles
3. **Scale Broadcasting**: Efficient scale application in registers
4. **Packed Operations**: Process multiple low-bit values per instruction

## Performance Considerations

- **Memory Savings**: INT8 = 2x, INT4 = 4x over FP16
- **Compute Speedup**: INT8 tensor cores are 2x faster than FP16
- **Accuracy Trade-offs**: Per-group quantization improves accuracy at cost of complexity
- **Calibration**: Static quantization requires representative calibration data

## Common Patterns

```python
# Per-token INT8 quantization
x_q, scale = per_token_quant_int8(x)

# Block-wise quantized matmul
# A: [M, K] INT8, As: [M, K//block_k] scales
# B: [K, N] INT8, Bs: [K//block_k, N//block_n] scales
C = w8a8_block_int8_matmul(A, B, As, Bs, block_size=[block_n, block_k])

# AWQ dequantization
# qweight: packed 4-bit [N, K//8] int32
weight = awq_dequantize(qweight, scales, zeros, group_size=128)
```

## References

- [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)
- [GPTQ: Accurate Post-Training Quantization](https://arxiv.org/abs/2210.17323)
- [LLM.int8(): 8-bit Matrix Multiplication](https://arxiv.org/abs/2208.07339)
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)
