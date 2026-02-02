# Normalization Kernel Analysis

## Overview

Normalization kernels stabilize training and inference by normalizing activations. Modern LLMs primarily use RMSNorm (simpler, faster) instead of LayerNorm. Gated variants are used in architectures like Mamba.

## Mathematical Foundations

### Layer Normalization (LayerNorm)

For input $x \in \mathbb{R}^D$:
$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

where:
- $\mu = \frac{1}{D}\sum_i x_i$ (mean)
- $\sigma^2 = \frac{1}{D}\sum_i (x_i - \mu)^2$ (variance)
- $\gamma, \beta \in \mathbb{R}^D$ (learnable scale and shift)
- $\epsilon$ is a small constant for numerical stability

### RMS Normalization (RMSNorm)

Simplifies LayerNorm by removing mean centering:
$$\text{RMSNorm}(x) = \gamma \cdot \frac{x}{\sqrt{\frac{1}{D}\sum_i x_i^2 + \epsilon}}$$

Or equivalently:
$$\text{RMSNorm}(x) = \gamma \cdot \frac{x}{\text{RMS}(x)}$$

where $\text{RMS}(x) = \sqrt{\frac{1}{D}\sum_i x_i^2 + \epsilon}$

### Gated RMSNorm

Used in Mamba architectures:
$$\text{GatedRMSNorm}(x, z) = \text{RMSNorm}(x) \odot \text{SiLU}(z)$$

Two variants:
1. **Norm before gate**: $\text{RMSNorm}(x) \odot \text{SiLU}(z)$
2. **Norm after gate**: $\text{RMSNorm}(x \odot \text{SiLU}(z))$

### Gated LayerNorm

Similarly for LayerNorm:
$$\text{GatedLayerNorm}(x, z) = \text{LayerNorm}(x) \odot \text{SiLU}(z)$$

## Kernel Implementations

### layernorm_gated_mamba.py
- **Source**: Mamba SSM
- **Functions**: `_layer_norm_fwd_1pass_kernel`, `_layer_norm_bwd_kernel`
- **Features**:
  - Single-pass mean and variance computation
  - Fused gating with SiLU activation
  - Supports both LayerNorm and RMSNorm
  - Welford's online algorithm for numerical stability

### Forward Kernel (Single-Pass)

Computes mean and variance in one pass using Welford's algorithm:
```python
# Welford's online algorithm
mean = 0.0
M2 = 0.0
for i in range(D):
    delta = x[i] - mean
    mean += delta / (i + 1)
    M2 += delta * (x[i] - mean)
variance = M2 / D
```

### Backward Kernel

For RMSNorm backward:
$$\frac{\partial L}{\partial x_i} = \frac{\gamma_i}{\text{RMS}} \left( \frac{\partial L}{\partial y_i} - \frac{x_i}{D \cdot \text{RMS}^2} \sum_j \gamma_j x_j \frac{\partial L}{\partial y_j} \right)$$

## Numerical Considerations

### Variance Computation

Two methods:
1. **Two-pass**: Compute mean first, then variance (more stable)
2. **Welford's**: Single-pass online algorithm (memory efficient)

Welford's is preferred for Triton as it avoids two memory reads.

### Small Epsilon

Typical values: $\epsilon = 10^{-5}$ to $10^{-6}$

Too small → numerical instability
Too large → affects normalization quality

### Gradient Flow

RMSNorm has better gradient flow than LayerNorm for very deep networks because it doesn't center the distribution.

## Performance Considerations

1. **Single-Pass**: Avoid reading input twice for mean and variance
2. **Fused Operations**: Combine norm with preceding/following operations
3. **Block Size**: Match to hidden dimension for efficient parallelization
4. **Memory Access**: Coalesced reads across the hidden dimension

## Common Patterns

```python
# RMSNorm
variance = x.pow(2).mean(-1, keepdim=True)
x_norm = x * torch.rsqrt(variance + eps)
output = x_norm * weight

# Gated RMSNorm (Mamba style)
x_norm = rmsnorm(x, weight)
output = x_norm * F.silu(z)

# Pre-norm transformer block
x = x + attention(rmsnorm(x))
x = x + ffn(rmsnorm(x))
```

## RMSNorm vs LayerNorm

| Aspect | LayerNorm | RMSNorm |
|--------|-----------|---------|
| Mean centering | Yes | No |
| Bias parameter | Yes | No |
| Compute cost | Higher | Lower (~20% faster) |
| Memory | More | Less |
| LLM adoption | GPT-2/3 | LLaMA, Mistral, etc. |

## References

- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- [Layer Normalization](https://arxiv.org/abs/1607.06450)
- [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752)
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
