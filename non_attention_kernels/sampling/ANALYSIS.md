# Sampling Kernel Analysis

## Overview

Sampling kernels handle the probabilistic token generation process in LLM inference. These kernels transform model logits into token selections through various strategies like temperature scaling, top-k/top-p filtering, and penalty application.

## Mathematical Foundations

### Temperature Scaling

Temperature controls the randomness of the distribution:
$$\text{scaled\_logits} = \frac{\text{logits}}{T}$$

- $T < 1$: Sharper distribution (more deterministic)
- $T > 1$: Flatter distribution (more random)
- $T = 1$: Original distribution

### Top-K Filtering

Keep only the $k$ highest probability tokens:
$$\text{filtered}_i = \begin{cases} \text{logits}_i & \text{if } i \in \text{top\_k\_indices} \\ -\infty & \text{otherwise} \end{cases}$$

### Top-P (Nucleus) Filtering

Keep tokens whose cumulative probability exceeds threshold $p$:
1. Sort tokens by probability in descending order
2. Compute cumulative sum of probabilities
3. Keep tokens until cumulative sum reaches $p$

$$\text{filtered}_i = \begin{cases} \text{logits}_i & \text{if } \sum_{j \leq i} P_j \leq p \\ -\infty & \text{otherwise} \end{cases}$$

### Min-P Filtering

Keep tokens with probability at least $\text{min\_p} \times \max(P)$:
$$\text{filtered}_i = \begin{cases} \text{logits}_i & \text{if } P_i \geq \text{min\_p} \cdot \max_j(P_j) \\ -\infty & \text{otherwise} \end{cases}$$

### Repetition Penalty

Penalize tokens that appeared in the context:
$$\text{logits}'_i = \begin{cases} \text{logits}_i / \alpha & \text{if } \text{logits}_i > 0 \text{ and } i \in \text{context} \\ \text{logits}_i \cdot \alpha & \text{if } \text{logits}_i \leq 0 \text{ and } i \in \text{context} \\ \text{logits}_i & \text{otherwise} \end{cases}$$

### Gumbel-Max Sampling

Sample from categorical distribution using Gumbel trick:
$$\text{token} = \arg\max_i (\text{logits}_i + G_i)$$

where $G_i \sim \text{Gumbel}(0, 1) = -\log(-\log(U_i))$ and $U_i \sim \text{Uniform}(0, 1)$

## Kernel Implementations

### temperature_vllm.py
- Applies per-token temperature scaling
- Grid: one block per token

### min_p_vllm.py
- Dynamic probability threshold filtering
- Computes max probability per row efficiently

### penalties_vllm.py
- Repetition, frequency, and presence penalties
- Uses token counting for efficient penalty lookup

### gumbel_vllm.py
- Gumbel-Max trick for efficient sampling
- Parallelized random number generation

### logprob_vllm.py
- Computes log probabilities for selected tokens
- Efficient log-softmax with numerical stability

### logit_bias_vllm.py
- Applies sparse logit biases
- Efficient for small bias vocabularies

## Speculative Decoding

### rejection_sample_vllm.py
- Draft model generates candidates
- Target model verifies via rejection sampling
- Accept probability: $\min(1, P_{\text{target}}(x) / P_{\text{draft}}(x))$

### eagle_utils_vllm.py
- EAGLE speculative decoding implementation
- Tree-structured draft generation

## Performance Considerations

1. **Memory Bandwidth**: Sampling is often memory-bound
2. **Numerical Stability**: Use log-space for probabilities
3. **Batch Heterogeneity**: Different sequences may use different parameters
4. **Random Number Quality**: Use high-quality PRNGs for reproducibility

## Common Patterns

```python
# Temperature + Top-k + Top-p sampling pipeline
logits = apply_temperature(logits, temperature)
logits = apply_top_k(logits, top_k)
logits = apply_top_p(logits, top_p)
logits = apply_repetition_penalty(logits, input_ids, penalty)
tokens = gumbel_sample(logits)
```

## References

- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)
- [EAGLE: Speculative Sampling](https://arxiv.org/abs/2401.15077)
- [Contrastive Decoding](https://arxiv.org/abs/2210.15097)
