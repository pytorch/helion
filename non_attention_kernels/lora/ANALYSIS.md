# LoRA (Low-Rank Adaptation) Kernel Analysis

## Overview

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that trains low-rank decomposition matrices instead of full weight updates. LoRA kernels enable efficient inference with multiple adapters by batching low-rank operations.

## Mathematical Foundations

### Low-Rank Decomposition

Instead of updating the full weight matrix $W \in \mathbb{R}^{d_{out} \times d_{in}}$, LoRA trains two low-rank matrices:
- $A \in \mathbb{R}^{r \times d_{in}}$ (shrink/down-projection)
- $B \in \mathbb{R}^{d_{out} \times r}$ (expand/up-projection)

where $r \ll \min(d_{in}, d_{out})$ is the rank.

### Forward Pass

Original: $y = xW^T$

With LoRA:
$$y = xW^T + \alpha \cdot x A^T B^T = x(W + \alpha BA)^T$$

where $\alpha = \frac{\alpha_{\text{lora}}}{r}$ is the scaling factor.

### Two-Stage Computation

1. **Shrink (A)**: Project to low-rank space
   $$h = xA^T \in \mathbb{R}^{B \times r}$$

2. **Expand (B)**: Project back to output space
   $$\Delta y = hB^T \cdot \alpha \in \mathbb{R}^{B \times d_{out}}$$

### Batched LoRA

For serving multiple LoRA adapters simultaneously:
$$y_i = x_i W^T + \sum_{j \in \text{adapters}_i} \alpha_j \cdot x_i A_j^T B_j^T$$

Each sample can use a different adapter (or combination).

## Kernel Implementations

### lora_shrink_vllm.py
- **Function**: `_lora_shrink_kernel`
- **Operation**: $h = xA^T$ (batched)
- **Features**:
  - SGMV (Segment GEMV) batching
  - Efficient for small rank dimensions
  - Handles variable-length sequences

### lora_expand_vllm.py
- **Function**: `_lora_expand_kernel`
- **Operation**: $y += hB^T \cdot \alpha$ (with accumulation)
- **Features**:
  - Add-to-output for multiple adapters
  - Scale factor application
  - Column-major optimization for B matrix

### SGLang Variants

- **chunked_sgmv_expand/shrink**: Chunked versions for large batches
- **qkv_lora_b**: Specialized for QKV projection (3 outputs)
- **gate_up_lora_b**: Specialized for MLP gate+up projection (2 outputs)
- **sgemm_lora_a/b**: Full SGEMM-based LoRA

## Optimization Strategies

1. **Segment Batching**: Group samples by adapter for efficient GEMM
2. **Fused Operations**: Combine shrink+expand when possible
3. **Weight Stacking**: Stack A/B matrices for all adapters
4. **Column-Major B**: Optimize memory access pattern for B matrix

## Memory Layout

```
# A matrices stacked: [num_adapters, rank, hidden_dim]
# B matrices stacked: [num_adapters, output_dim, rank]
# Indices per sample: [batch_size] -> adapter index (-1 = no adapter)
```

## Performance Considerations

- **Rank Matters**: Higher rank = more compute but better quality
- **Adapter Count**: More adapters = larger memory footprint
- **Batching Efficiency**: Mixed adapters reduce batching benefits
- **Memory Bandwidth**: Low-rank operations are often memory-bound

## Common Patterns

```python
# Single adapter forward
intermediate = x @ lora_a.T  # Shrink: [batch, hidden] -> [batch, rank]
delta = intermediate @ lora_b.T * scale  # Expand: [batch, rank] -> [batch, output]
output = base_output + delta

# Multi-adapter forward
for i, (adapter_idx, x_i) in enumerate(zip(indices, x)):
    if adapter_idx >= 0:
        h = x_i @ lora_a_stacked[adapter_idx].T
        output[i] += h @ lora_b_stacked[adapter_idx].T * scale
```

## Typical Configurations

| Layer Type | Rank | Alpha | Params (7B model) |
|------------|------|-------|-------------------|
| Q/K/V      | 8    | 16    | ~2M per layer    |
| Q/K/V      | 32   | 64    | ~8M per layer    |
| All linear | 16   | 32    | ~12M per layer   |

## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [S-LoRA: Serving Thousands of Concurrent LoRA Adapters](https://arxiv.org/abs/2311.03285)
- [Punica: Multi-Tenant LoRA Serving](https://arxiv.org/abs/2310.18547)
