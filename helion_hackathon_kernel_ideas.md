# Helion Hackathon Kernel Ideas

A curated list of GPU kernel ideas for the Helion hackathon, focusing on kernel speed performance competition.

---

## FlashInfer Kernel Catalog (Primary Reference)

FlashInfer is the MLSys 2025 Best Paper award-winning kernel library for LLM inference. Below is the complete catalog of FlashInfer kernels organized by category.

### Attention Kernels

| Kernel | Description | Key API |
|--------|-------------|---------|
| **Single Decode** | Single-request decode attention | `single_decode_with_kv_cache` |
| **Batch Decode** | Batched decode with paged KV cache | `batch_decode_with_kv_cache` |
| **Prefill Attention** | Initial sequence processing | `single_prefill_with_kv_cache` |
| **Append Attention** | Incremental sequence extension | Append APIs |
| **MLA Attention** | DeepSeek Multi-Latent Attention | MLA-specific decode/prefill |
| **Cascade Attention** | Hierarchical KV-Cache for shared prefixes | `merge_state`, `merge_states` |
| **POD-Attention** | Fused prefill+decode for mixed batching | POD APIs | [POD-Attention Paper](https://arxiv.org/html/2410.18038v1)
| **Block-Sparse Attention** | Structured sparsity patterns | Block sparse APIs |
| **Variable Block-Sparse** | Flexible sparse configurations | Variable block APIs |

- **Reference**: [FlashInfer Attention Docs](https://docs.flashinfer.ai/)

### Sampling Kernels (Sorting-Free)

| Kernel | Description | Algorithm |
|--------|-------------|-----------|
| **Inverse Transform Sampling** | CDF-based token selection | BlockReduce + BlockScan + early stopping |
| **Rejection Sampling** | Threshold-based filtering | Iterative pivot refinement |
| **Dual-Pivot Rejection Sampling** | O(log(1/ε)) guaranteed convergence | Binary search with two pivots |
| **Top-K Sampling** | Keep K highest probability tokens | Sorting-free parallel selection |
| **Top-P (Nucleus) Sampling** | Cumulative probability threshold | `top_p_renorm_probs` |
| **Min-P Sampling** | Filter below `p_base × p_max` | `min_p_sampling_from_probs` |
| **Combined Top-K/Top-P** | Composable filtering strategies | `top_k_top_p_sampling_from_probs` |
| **Chain Speculative Sampling** | Speculative decoding verification | Chain sampling APIs |

**Performance**: 50%+ reduction in sampling time vs sorting-based PyTorch implementations.

- **Reference**: [FlashInfer Sampling Blog](https://flashinfer.ai/2025/03/10/sampling.html)

### Normalization Kernels

| Kernel | Description |
|--------|-------------|
| **Fused RMSNorm + FP8 Quantize** | Combined norm + quantization |

### Paging & KV-Cache Kernels

| Kernel | Description |
|--------|-------------|
| **Page Construction** | KV-cache page allocation |
| **Top-K Page Construction** | Fused top-k + page building |
| **Paged KV-Cache Management** | Memory-efficient cache ops |
| **Ragged KV-Cache** | Variable-length sequences |

---

## MLSys 2026 FlashInfer Contest Tracks

The [MLSys 2026 FlashInfer Contest](https://mlsys26.flashinfer.ai/) defines three competition tracks directly relevant to this hackathon:

### Track A: Fused MoE + FP8

**Kernel**: Fused Mixture-of-Experts with FP8 quantization support

- **Challenge**: Optimize routing + grouped GEMM fusion with FP8 precision
- **Key techniques**: Online FP8 quantization, pipelined group GEMM, align & sort fusion
- **Models**: DeepSeek-V3 (256 experts), Llama-4

### Track B: DeepSeek Sparse Attention

**Kernels**:
- `dsa_topk_indexer_fp8_h64_d128_topk256_ps64` - Token selection indexer
- `dsa_sparse_attention_h16_ckv512_kpe64_topk256_ps64` - Sparse attention computation

- **Challenge**: Implement production sparse attention from DeepSeek V3.2
- **Key techniques**: Top-K token selection, hierarchical sparsity, FP8 KV cache

### Track C: Gated DeltaNet (Linear Attention)

**Kernels**:
- `gdn_decode_qk16_v32_d128_k_last` - Decode kernel
- `gdn_prefill_qk16_v32_d128_k_last` - Prefill kernel

- **Challenge**: Implement Qwen3-Next's linear attention variant
- **Key techniques**: Chunkwise parallelism, gated delta rule, fast-weight RNN

---

## FlashInfer-Bench Evaluated Workloads

The [FlashInfer-Bench](https://flashinfer.ai/2025/10/21/flashinfer-bench.html) platform evaluates kernels on real LLM workloads:

| Kernel Family | Models Tested |
|---------------|---------------|
| **GQA Paged Attention** | All decoder models |
| **Sampling** | All autoregressive models |

---

### 2. Attention Mechanism Variants

#### Multi-Head Latent Attention (MLA)
- **Description**: Low-rank KV cache compression (64x smaller footprint), used in DeepSeek-V2/V3
- **Key techniques**: Joint low-rank decomposition of K/V projections, compressed latent caching
- **Performance**: 488GB → ~7.6GB KV cache for 128K context
- **References**:
  - [FlashMLA (Official)](https://github.com/deepseek-ai/FlashMLA) - Hopper kernel achieving 3000 GB/s, 580 TFLOPS
  - [DeepSeek-V3 MLA PR](https://github.com/deepseek-ai/DeepSeek-V3/pull/684)
  - [TransMLA Paper](https://arxiv.org/abs/2502.07864) - 10.6x inference speedup
  - [MLA Explainer](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html)

#### Gated Attention
- **Description**: Query-dependent sparse gate after SDPA output, eliminating attention sink phenomenon
- **Key techniques**: Sigmoid gating per head, non-linearity in low-rank transformation
- **Performance**: Attention sink reduced from 46.7% → 4.8%, enables larger learning rates
- **References**:
  - [Official Implementation](https://github.com/qiuzh20/gated_attention) - NeurIPS 2025 Best Paper
  - [Qwen Paper](https://towardsdatascience.com/neurips-2025-best-paper-review-qwens-systematic-exploration-of-attention-gating/)

#### Differential Attention (DIFF Transformer)
- **Description**: Computes attention as difference of two softmax maps to cancel noise
- **Key techniques**: Dual attention map computation, sparse pattern emergence via subtraction
- **Performance**: 35-40% fewer parameters/tokens needed vs standard Transformer
- **References**:
  - [ICLR 2025 Paper](https://arxiv.org/abs/2410.05258)
  - [DiffCLIP Implementation](https://github.com/hammoudhasan/DiffCLIP)

---

### 3. Linear Attention Variants

#### Gated DeltaNet
- **Description**: Linear attention with gated delta rule for improved memory management
- **Key techniques**: Chunkwise parallelism, data-dependent gating, fast-weight RNN formulation
- **Performance**: Used in Qwen3-Next, competitive with full attention at lower cost
- **References**:
  - [NVLabs Official](https://github.com/NVlabs/GatedDeltaNet) - ICLR 2025
  - [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) - Triton kernels

#### RetNet / RWKV-6 / RWKV-7
- **Description**: Recurrent linear attention with exponential decay (RetNet) or receptance-weighted key-value (RWKV)
- **Key techniques**: Constant-size cache, parallel/chunked processing
- **Performance**: RWKV-X achieves 1.37x speedup over FA3 at 128K tokens
- **References**:
  - [Flash Linear Attention Library](https://github.com/fla-org/flash-linear-attention)
  - [RWKV Survey](https://arxiv.org/pdf/2412.14847)

#### Mamba / Mamba-2
- **Description**: State-space models as input-conditioned RNNs with selective gating
- **Key techniques**: Hardware-aware parallel scan, selective state propagation
- **Performance**: 5x higher throughput than Transformers, linear scaling
- **References**:
  - [Mamba Paper](https://arxiv.org/pdf/2312.00752)
  - [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)

---

### 4. Sparse Attention Variants

#### Native Sparse Attention (NSA)
- **Description**: Hardware-aligned sparse attention with hierarchical token modeling
- **Key techniques**: Compressed coarse tokens + selective fine tokens + sliding window
- **References**:
  - [NSA Paper](https://arxiv.org/pdf/2502.11089)
  - [FSA (faster alternative)](https://arxiv.org/pdf/2508.18224) - 7.6x speedup on token selection

#### Block Sparse Attention
- **Description**: Mixed sparse patterns (streaming, block granularity)
- **References**:
  - [MIT HAN Lab Implementation](https://github.com/mit-han-lab/Block-Sparse-Attention) - Based on FlashAttention 2.4.2
  - [BLASST Paper](https://www.arxiv.org/pdf/2512.12087) - Dynamic pruning via softmax thresholding

#### Neighborhood Attention (NATTEN)
- **Description**: Multi-dimensional sliding window attention with stride optimization
- **Key techniques**: Strided neighborhood attention, generalized NA
- **Performance**: Up to 11.1x speedup with optimal strides
- **References**:
  - [NATTEN Library](https://github.com/SHI-Labs/NATTEN) - SM50 to SM100/103 kernels
  - [GNA Paper](https://arxiv.org/html/2504.16922v1)

---

### 5. Quantized Attention Kernels

#### SageAttention (INT8/FP8/FP4)
- **Description**: Quantized QK^T (INT8) and PV (FP8) attention
- **Key techniques**: Per-block quantization, smoothing for Q/K matrices
- **Performance**: 2-5x speedup vs FlashAttention; SageAttention3 achieves 1038 TOPS with FP4 on RTX5090
- **References**:
  - [SageAttention](https://github.com/thu-ml/SageAttention) - ICLR/ICML/NeurIPS 2025
  - [SageAttention3 Paper](https://arxiv.org/html/2505.11594v3) - INT8 training support
  - [ROCm Port](https://github.com/EmbeddedLLM/SageAttention-rocm)

---

## Additional Resources

- [FlashInfer GitHub](https://github.com/flashinfer-ai/flashinfer) - Official kernel library
- [FlashInfer Documentation](https://docs.flashinfer.ai/) - API reference
- [FlashInfer-Bench](https://flashinfer.ai/2025/10/21/flashinfer-bench.html) - Benchmarking platform
- [MLSys 2026 Contest](https://mlsys26.flashinfer.ai/) - Competition tracks
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) - Linear attention implementations
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel) - Fused training kernels
- [vLLM](https://github.com/vllm-project/vllm) - Reference implementations
- [SGLang](https://github.com/sgl-project/sglang) - MoE and attention kernels

### Megakernel Resources

- [Mirage MPK GitHub](https://github.com/mirage-project/mirage) - Automatic megakernel compiler
- [Mirage Paper](https://arxiv.org/abs/2512.22219) - Technical details
- [Hazy Research Megakernels](https://github.com/HazyResearch/Megakernels) - Llama megakernels
- [ThunderKittens](https://github.com/HazyResearch/ThunderKittens) - Tile primitives for kernels
- [Llama-1B Megakernel Blog](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) - Low-latency design
- [Llama-70B TP Blog](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main) - Tensor-parallel megakernel
- [ThunderMLA Blog](https://hazyresearch.stanford.edu/blog/2025-03-04-thundermla) - Fused MLA decode
- [ParallelKittens Blog](https://hazyresearch.stanford.edu/blog/2025-11-17-pk) - Multi-GPU kernels
