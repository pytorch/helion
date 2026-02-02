# MoE (Mixture of Experts) Kernel Analysis

## Overview

Mixture of Experts (MoE) is a neural network architecture that uses a gating mechanism to route inputs to specialized "expert" sub-networks. MoE kernels are critical for efficient inference of large models like Mixtral, DeepSeek-V3, and Llama-4.

## Mathematical Foundations

### Token Routing

Given input tokens $x \in \mathbb{R}^{B \times D}$ and router weights $W_r \in \mathbb{R}^{E \times D}$:

$$\text{logits} = x W_r^T$$
$$\text{probs} = \text{softmax}(\text{logits})$$
$$\text{top\_k\_indices}, \text{top\_k\_weights} = \text{topk}(\text{probs}, k)$$

The routing weights are typically renormalized:
$$\text{top\_k\_weights} = \frac{\text{top\_k\_weights}}{\sum \text{top\_k\_weights}}$$

### Expert Computation

Each expert performs a standard FFN computation:
$$\text{output}_e = \text{activation}(x W_1^{(e)}) \cdot W_2^{(e)}$$

For SwiGLU activation (common in modern LLMs):
$$\text{output}_e = \text{SiLU}(x W_{\text{gate}}^{(e)}) \odot (x W_{\text{up}}^{(e)}) \cdot W_{\text{down}}^{(e)}$$

### Final Output

$$y = \sum_{e \in \text{top\_k}} w_e \cdot \text{output}_e$$

## Kernel Implementations

### fused_moe_vllm.py
- **Source**: vLLM
- **Key Functions**: `fused_moe`, `fused_moe_kernel`
- **Features**:
  - Fused expert routing and computation
  - Support for FP8 quantization
  - GPTQ/AWQ quantized MoE
  - Efficient token permutation

### fused_moe_sglang.py
- **Source**: SGLang
- **Key Functions**: `fused_moe_kernel`, `act_and_mul_kernel`
- **Features**:
  - Optimized for SGLang's batching strategy
  - Fused SiLU+Mul activation

## Optimization Strategies

1. **Token Grouping**: Group tokens by their assigned experts to maximize memory coalescing
2. **Expert Parallelism (EP)**: Distribute experts across multiple GPUs
3. **Fused Operations**: Combine routing, permutation, GEMM, and activation in single kernels
4. **Block Alignment**: Align token counts to GEMM tile sizes for efficiency

## Performance Considerations

- **Load Balancing**: Uneven expert assignment leads to stragglers
- **Memory Bandwidth**: Expert weights are large; minimize redundant loads
- **Communication**: EP requires all-to-all communication overhead
- **Quantization**: FP8/INT8 MoE significantly reduces memory and compute

## Common Patterns

```python
# Token routing
routing_weights = softmax(hidden_states @ router_weight.T)
topk_weights, topk_indices = topk(routing_weights, k=2)

# Fused MoE forward
output = fused_moe(
    hidden_states,  # [batch, hidden_dim]
    w1,             # [num_experts, intermediate_dim*2, hidden_dim] (gate+up)
    w2,             # [num_experts, hidden_dim, intermediate_dim] (down)
    topk_weights,
    topk_indices
)
```

## References

- [Mixtral of Experts](https://arxiv.org/abs/2401.04088)
- [DeepSeekMoE](https://arxiv.org/abs/2401.06066)
- [Switch Transformers](https://arxiv.org/abs/2101.03961)
