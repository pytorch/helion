# Attention Kernels Used by Modern LLMs (2025 Models Only)

This document maps attention mechanisms to models released on or after January 1, 2025, based on extensive research of technical reports and architecture analyses.

---

## Quick Reference Table

| Attention Type | Models Using It (2025+) |
|---------------|-------------------------|
| **GQA** | Llama 4, Qwen3, Gemma 3, Mistral Large 3, Codestral 25.01 |
| **MLA** | DeepSeek R1, Kimi K2 |
| **SWA** | Gemma 3 |
| **KDA** | Kimi Linear |
| **Gated DeltaNet** | Qwen3-Next |
| **NSA** | DeepSeek (research paper) |

---

## 1. Grouped-Query Attention (GQA)

**Concept**: Multiple query heads share fewer key-value heads, reducing KV cache memory while maintaining most of MHA's expressiveness.

### Models Using GQA (2025+)

| Model Family | Version | Release Date | GQA Details | Context Length |
|-------------|---------|--------------|-------------|----------------|
| **Llama** | Llama 4 | Apr 5, 2025 | GQA + MoE, all sizes | 10M tokens |
| **Qwen** | Qwen3 | Apr 28, 2025 | GQA + QK-Norm (removed QKV-bias) | 128K |
| **Gemma** | Gemma 3 | Mar 12, 2025 | GQA + 5:1 local/global SWA | 128K |
| **Mistral** | Mistral Large 3 | Dec 2, 2025 | GQA + MoE (675B total, 41B active) | 128K |
| | Codestral 25.01 | Jan 2025 | GQA, code-optimized | 256K |

### Technical Notes
- GQA typically uses 8 KV heads shared across 32+ query heads
- Memory reduction: ~1.5-2x compared to standard MHA
- Remains the dominant attention mechanism in 2025 models

---

## 2. Multi-Head Latent Attention (MLA)

**Concept**: Compresses K and V tensors into a low-dimensional latent space before caching. During inference, decompress back to full size. Achieves better compression than GQA while maintaining MHA-level expressiveness.

### Models Using MLA (2025+)

| Model Family | Version | Release Date | MLA Details | Compression |
|-------------|---------|--------------|-------------|-------------|
| **DeepSeek** | DeepSeek R1 | Jan 20, 2025 | MLA + MoE, 671B total, 37B active | 28x smaller KV cache |
| **Kimi** | Kimi K2 | Jul 2025 | MLA (same design as DeepSeek), 1.04T total, 32B active | ~28x |

### Technical Notes
- DeepSeek R1 uses d_h=128, H=128, d_c=512 for 32x compression ratio
- Joint KV compression: Single down-projection matrix for both K and V
- KV cache reduced from ~213 GB to ~7.6 GB
- RoPE handled via decoupled embeddings (separate rope_k dimension)
- MLA is strictly more expressive than GQA (GQA can be represented as MLA)

### Key Innovation
MLA trades one-time compute during training for repeated efficiency gains at inference. The latent space (512 dims) captures the essential semantics that would otherwise require 14K+ dimensions.

---

## 3. Sliding Window Attention (SWA)

**Concept**: Each token only attends to a fixed window of preceding tokens. Information flows through stacked layers to access longer context implicitly.

### Models Using SWA (2025+)

| Model | Release Date | SWA Configuration | Global/Local Ratio |
|-------|--------------|-------------------|-------------------|
| **Gemma 3** | Mar 12, 2025 | 1024-token window | 5:1 (5 local : 1 global) |

### Technical Notes
- Gemma 3's 5:1 ratio reduces attention compute by ~5x
- Gemma 3 combines GQA + SWA: KV cache reduced to 0.78 GB (from 17 GB with MHA)

---

## 4. Kimi Delta Attention (KDA)

**Concept**: Linear attention variant that extends Gated DeltaNet with channel-wise forget gates. Enables O(n) scaling with sequence length.

### Models Using KDA (2025+)

| Model | Release Date | Architecture | Performance |
|-------|--------------|--------------|-------------|
| **Kimi Linear** | Oct 2025 | 3:1 KDA:Global ratio | 6.3x decoding throughput |

### Technical Details
- Extends Gated DeltaNet with **channel-wise** (not head-wise) forget gates
- Each feature dimension has independent forgetting rate
- 75% KV cache reduction vs full attention
- Works especially well for 1M+ token contexts
- Uses hybrid 3:1 ratio (3 KDA blocks : 1 global attention block) to address long-range retrieval weakness

### Key Innovation
Channel-wise gating allows fine-grained control over which information dimensions are retained vs forgotten, providing more expressiveness than head-wise approaches.

---

## 5. Gated DeltaNet

**Concept**: Linear attention with gated memory updates. Uses delta rule for efficient recurrent computation while maintaining expressiveness.

### Models Using Gated DeltaNet (2025+)

| Model | Release Date | Architecture | Details |
|-------|--------------|--------------|---------|
| **Qwen3-Next** | Sep 10, 2025 | 3:1 linear:global ratio | 80B total, 3B active |

### Technical Details
- Gated linear attention with delta rule updates
- Combines with full attention in 3:1 hybrid ratio
- Addresses linear attention's weakness with long-range retrieval while maintaining O(n) efficiency
- Base mechanism that KDA extends with channel-wise gating

### Hybrid Architecture Pattern
Uses 3 Gated DeltaNet blocks followed by 1 global attention block. This pattern is shared with Kimi Linear (KDA), representing an emerging standard for efficient long-context models.

---

## 6. Native Sparse Attention (NSA)

**Concept**: Hardware-aligned sparse attention that's natively trainable. Combines multiple attention paths for efficiency.

### Research

| Source | Release Date | Status | Details |
|--------|--------------|--------|---------|
| **DeepSeek NSA** | Feb 16, 2025 | Research paper | Won ACL 2025 Best Paper |

### Three-Branch Architecture
1. **Compressed (Coarse-grained)**: Summarizes context blocks
2. **Selected (Fine-grained)**: Picks critical tokens
3. **Sliding (Local)**: Recent context window

### Technical Notes
- Designed for 64K+ sequences
- Surpasses full attention on long-context benchmarks
- Foundation for DeepSeek Sparse Attention (DSA)

---

## Model-Centric Summary (2025+ Only)

### DeepSeek
| Version | Release Date | Details |
|---------|--------------|---------|
| DeepSeek R1 | Jan 20, 2025 | MLA + MoE, 671B total/37B active, RL-trained reasoning |

### Qwen
| Version | Release Date | Details |
|---------|--------------|---------|
| Qwen3-Next | Sep 10, 2025 | Gated DeltaNet (3:1 hybrid), 80B total/3B active |
| Qwen3 | Apr 28, 2025 | GQA + QK-Norm, 0.6B-235B sizes, 128K context |

### Kimi/Moonshot
| Version | Release Date | Details |
|---------|--------------|---------|
| Kimi Linear | Oct 2025 | KDA (channel-wise gated linear), 48B/3B active, 3:1 hybrid |
| Kimi K2 | Jul 2025 | MLA, 1.04T/32B active |
| Kimi K1.5 | Jan 20, 2025 | Matched OpenAI o1 performance |

### Gemma
| Version | Release Date | Details |
|---------|--------------|---------|
| Gemma 3 | Mar 12, 2025 | GQA + SWA 5:1, 1B-27B sizes, 128K context |

### Llama
| Version | Release Date | Details |
|---------|--------------|---------|
| Llama 4 | Apr 5, 2025 | GQA + MoE, 10M context |

### Mistral
| Version | Release Date | Details |
|---------|--------------|---------|
| Mistral Large 3 | Dec 2, 2025 | GQA + MoE, 675B total/41B active |
| Codestral 25.01 | Jan 2025 | GQA, 256K context for code |

### OpenAI GPT
| Version | Release Date | Details |
|---------|--------------|---------|
| GPT-5 | Aug 7, 2025 | Auto-routing between fast/reasoning modes, architecture undisclosed |

---

## 2025 Attention Evolution Timeline

```
Jan 2025:  DeepSeek R1 - MLA reasoning model
           Kimi K1.5 - Matched o1 performance
           Codestral 25.01 - GQA, 256K code context
Feb 2025:  DeepSeek NSA paper - Sparse attention research (ACL Best Paper)
Mar 2025:  Gemma 3 - GQA + SWA 5:1, 128K context
Apr 2025:  Llama 4 - GQA + MoE, 10M context
           Qwen3 - GQA + QK-Norm
Jul 2025:  Kimi K2 - MLA, 1T params
Aug 2025:  GPT-5 - Auto-routing architecture
Sep 2025:  Qwen3-Next - Gated DeltaNet (3:1 hybrid)
Oct 2025:  Kimi Linear - KDA (channel-wise gated linear)
Dec 2025:  Mistral Large 3 - 675B MoE
```

---

## Key Takeaways for Kernel Implementation (2025 Models)

1. **GQA remains dominant** - Llama 4, Qwen3, Gemma 3, Mistral Large 3 all use it
2. **MLA is the efficiency frontier** - DeepSeek R1 and Kimi K2 achieve 28x KV compression
3. **Hybrid linear attention emerging** - KDA (Kimi Linear) and Gated DeltaNet (Qwen3-Next) use 3:1 linear:global ratio
4. **SWA optimized for long context** - Gemma 3's 5:1 ratio dramatically reduces KV cache
5. **MoE is standard** - Llama 4, DeepSeek R1, Mistral Large 3, Kimi K2 all use mixture-of-experts

---

## Triton Reference Implementations

### GQA (Grouped-Query Attention)
| Repository | Description | URL |
|------------|-------------|-----|
| **vLLM Triton Unified Attention** | Production-ready GQA kernel with "packing" along query dimension for Tensor core efficiency | [vllm/triton_flash_attention.py](https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_flash_attention.py) |


### MLA (Multi-Head Latent Attention)
| Repository | Description | URL |
|------------|-------------|-----|
| **FlashMLA (Official)** | DeepSeek's official efficient MLA kernels, up to 660 TFlops on H800 | [deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA) |
| **SGLang Triton MLA** | Community Triton implementation for DeepSeek-V2 | [sgl-project/sglang PR #905](https://github.com/sgl-project/sglang/pull/905) |
| **mla-experiments** | Experiments on MLA variants with KV-cache reduction | [ambisinister/mla-experiments](https://github.com/ambisinister/mla-experiments) |

### SWA (Sliding Window Attention)
| Repository | Description | URL |
|------------|-------------|-----|
| **MaxLSB/flash-attn2** | FlashAttention-2 with SWA support (fwd + bwd pass) | [MaxLSB/flash-attn2](https://github.com/MaxLSB/flash-attn2) |
| **vLLM SWA Optimization** | Triton kernel with pruning for unused KV tiles outside window | [vllm PR #24390](https://github.com/vllm-project/vllm/pull/24390) |
| **ROCm/aotriton** | Generalized SWA with negative window values for causal masks | [ROCm/aotriton](https://github.com/ROCm/aotriton) |
| **Dao-AILab/flash-attention** | SWA contributed by Mistral AI (work in progress) | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) |

### KDA (Kimi Delta Attention)
| Repository | Description | URL |
|------------|-------------|-----|
| **Kimi-Linear (Official)** | Moonshot AI's official KDA kernel release | [MoonshotAI/Kimi-Linear](https://github.com/MoonshotAI/Kimi-Linear) |
| **flash-linear-attention** | KDA implementation with vLLM integration | [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) |

### Gated DeltaNet
| Repository | Description | URL |
|------------|-------------|-----|
| **GatedDeltaNet (Official NVIDIA)** | ICLR 2025 official implementation with hardware-efficient chunkwise algorithm | [NVlabs/GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet) |
| **flash-linear-attention** | Community Triton implementation with H100 optimizations | [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) |
| **LLMs-from-scratch** | Educational implementation with explanations | [rasbt/LLMs-from-scratch/deltanet](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/08_deltanet/README.md) |

### NSA (Native Sparse Attention)
| Repository | Description | URL |
|------------|-------------|-----|
| **native-sparse-attention (fla-org)** | Full training support with online top-k selection kernel | [fla-org/native-sparse-attention](https://github.com/fla-org/native-sparse-attention) |
| **native-sparse-attention-triton** | Efficient Triton impl, 29ms vs 810ms FlashAttn on 131K tokens | [XunhaoLai/native-sparse-attention-triton](https://github.com/XunhaoLai/native-sparse-attention-triton) |
| **nsa-impl** | PyTorch+Triton+FlexAttention implementation | [tilde-research/nsa-impl](https://github.com/tilde-research/nsa-impl) |
| **native-sparse-attention-pytorch** | Phil Wang's implementation | [lucidrains/native-sparse-attention-pytorch](https://github.com/lucidrains/native-sparse-attention-pytorch) |

---

## Sources

- [DeepSeek R1 Technical Report](https://arxiv.org/abs/2501.12948)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [Kimi Wikipedia](https://en.wikipedia.org/wiki/Kimi_(chatbot))
- [Kimi Linear Paper](https://arxiv.org/pdf/2510.26692)
- [Gemma 3 Explained - Google](https://developers.googleblog.com/gemma-explained-whats-new-in-gemma-3/)
- [Llama 4 Blog - Meta](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
- [NSA Paper](https://arxiv.org/abs/2502.11089)
- [GPT-5 Wikipedia](https://en.wikipedia.org/wiki/GPT-5)
