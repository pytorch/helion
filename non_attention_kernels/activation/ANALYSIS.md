# Activation Kernel Analysis

## Overview

Activation kernels implement non-linear functions used in neural networks. Modern LLMs primarily use gated activations like SwiGLU which combine element-wise multiplication with non-linear transformations.

## Mathematical Foundations

### SiLU (Swish)

Sigmoid Linear Unit:
$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

Properties:
- Smooth, non-monotonic
- Bounded below (~-0.278 at x ≈ -1.28)
- Unbounded above
- Derivative: $\text{SiLU}'(x) = \sigma(x)(1 + x(1 - \sigma(x)))$

### SwiGLU (Swish-Gated Linear Unit)

Combines SiLU with gating:
$$\text{SwiGLU}(x, y) = \text{SiLU}(x) \odot y = \frac{x}{1 + e^{-x}} \odot y$$

In transformer FFN:
$$\text{output} = \text{SwiGLU}(W_{\text{gate}} \cdot h, W_{\text{up}} \cdot h) \cdot W_{\text{down}}$$

### GeGLU (GELU-Gated Linear Unit)

Uses GELU instead of SiLU:
$$\text{GeGLU}(x, y) = \text{GELU}(x) \odot y$$

where $\text{GELU}(x) = x \cdot \Phi(x)$ and $\Phi$ is the standard normal CDF.

### Softplus

Smooth approximation to ReLU:
$$\text{Softplus}(x) = \log(1 + e^x)$$

Used in Mamba for positivity constraints.

## Kernel Implementations

### k_activations_mamba.py
- **Source**: Mamba SSM
- **Functions**: `_swiglu_fwd_kernel`, `_swiglu_bwd_kernel`, `swiglu`
- **Features**:
  - Fused SiLU and element-wise multiply
  - Autograd support for training
  - Multi-dtype support (FP32, FP16, BF16)

### Forward Kernel

```python
# Input: xy concatenated [batch, hidden_dim * 2]
# Split into gate (x) and up (y) projections
x, y = xy.chunk(2, dim=-1)
output = silu(x) * y
```

### Backward Kernel

Given gradient $\frac{\partial L}{\partial \text{output}}$:
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial \text{output}} \cdot y \cdot \text{SiLU}'(x)$$
$$\frac{\partial L}{\partial y} = \frac{\partial L}{\partial \text{output}} \cdot \text{SiLU}(x)$$

## Numerical Considerations

### SiLU Stability

For large positive $x$: $\text{SiLU}(x) \approx x$
For large negative $x$: $\text{SiLU}(x) \approx 0$

Gradient for large $|x|$:
- $x \to \infty$: $\text{SiLU}'(x) \to 1$
- $x \to -\infty$: $\text{SiLU}'(x) \to 0$

### Softplus Stability

For large positive $x$: $\text{Softplus}(x) \approx x$ (to avoid overflow)
For large negative $x$: $\text{Softplus}(x) \approx e^x$

## Performance Considerations

1. **Memory Fusion**: Combine activation with preceding/following operations
2. **Element-wise**: Purely element-wise operations are memory-bandwidth bound
3. **Chunking**: Process gate and up projections together to save memory traffic
4. **In-Place**: Consider in-place operations for large tensors

## Common Patterns

```python
# SwiGLU in transformer FFN
gate = hidden_states @ gate_proj.T
up = hidden_states @ up_proj.T
activated = swiglu(gate, up)  # or swiglu(torch.cat([gate, up], dim=-1))
output = activated @ down_proj.T

# Fused operation (single kernel)
gate_up = hidden_states @ gate_up_proj.T  # [batch, hidden_dim * 2]
activated = swiglu_fused(gate_up)  # [batch, hidden_dim]
output = activated @ down_proj.T
```

## Activation Functions Comparison

| Activation | Formula | Pros | Cons |
|------------|---------|------|------|
| ReLU | $\max(0, x)$ | Simple, fast | Dead neurons, not smooth |
| GELU | $x \cdot \Phi(x)$ | Smooth, better training | Slower than ReLU |
| SiLU/Swish | $x \cdot \sigma(x)$ | Smooth, self-gated | Slightly slower than GELU |
| SwiGLU | $\text{SiLU}(x) \cdot y$ | Best quality | 50% more params in FFN |

## References

- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- [Searching for Activation Functions](https://arxiv.org/abs/1710.05941) (Swish)
- [GELU Activation](https://arxiv.org/abs/1606.08415)
- [LLaMA: Open Foundation Models](https://arxiv.org/abs/2302.13971) (SwiGLU adoption)
