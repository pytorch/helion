# Top 58 Most Important Triton Kernels for LLM Inference

A curated selection of the most impactful and substantial Triton kernels from the unified kernel catalog, prioritized by production usage, architectural significance, and implementation complexity.

---

## Selection Criteria
- Production-critical (used in vLLM, SGLang)
- Powers cutting-edge architectures (MLA, NSA, KDA, Gated DeltaNet)
- Substantial implementation complexity
- Enables key optimizations (quantization, MoE, speculative decoding)

---

## Quick Reference Table

| # | Category | Kernel Name | Key Triton Code Pointer |
|---|----------|-------------|------------------------|
| 1 | Attention | **GQA Prefill** | [`vllm/.../triton_prefill_attention.py:36`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_prefill_attention.py#L36) `_fwd_kernel` |
| 2 | Attention | **GQA Decode Stage1** | [`vllm/.../triton_decode_attention.py:58`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_decode_attention.py#L58) `_fwd_kernel_stage1` |
| 3 | Attention | **GQA Decode Stage2** | [`vllm/.../triton_decode_attention.py:494`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_decode_attention.py#L494) `_fwd_kernel_stage2` |
| 4 | Attention | **Grouped Decode** | [`vllm/.../triton_decode_attention.py:248`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_decode_attention.py#L248) `_fwd_grouped_kernel_stage1` |
| 5 | Attention | **MLA Attention** | [`FlashMLA/benchmark/bench_flash_mla.py:135`](https://github.com/deepseek-ai/FlashMLA/blob/main/benchmark/bench_flash_mla.py#L135) `_mla_attn_kernel` |
| 6 | Attention | **MLA Softmax ReduceV** | [`FlashMLA/benchmark/bench_flash_mla.py:273`](https://github.com/deepseek-ai/FlashMLA/blob/main/benchmark/bench_flash_mla.py#L273) `_mla_softmax_reducev_kernel` |
| 7 | Attention | **SWA Forward** | [`flash-attn2/src/.../forward.py:21`](https://github.com/MaxLSB/flash-attn2/blob/main/src/flash_attention/forward.py#L21) `_attn_fwd`, `_attn_fwd_inner` |
| 8 | Attention | **SWA Backward** | [`flash-attn2/src/.../backward.py:76`](https://github.com/MaxLSB/flash-attn2/blob/main/src/flash_attention/backward.py#L76) `_attn_bwd_dk_dv`, `_attn_bwd_dq` |
| 9 | Attention | **Unified 2D** | [`vllm/.../triton_unified_attention.py:56`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_unified_attention.py#L56) `kernel_unified_attention_2d` |
| 10 | Attention | **Unified 3D (Segmented)** | [`vllm/.../triton_unified_attention.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_unified_attention.py) `kernel_unified_attention_3d` |
| 11 | Attention | **Chunked Prefill Paged Decode** | [`vllm/.../chunked_prefill_paged_decode.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/chunked_prefill_paged_decode.py) `chunked_prefill_paged_decode_kernel` |
| 12 | Attention | **Cascade State Merge** | [`flashinfer/.../cascade.py:24`](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/triton/kernels/cascade.py#L24) `merge_state_kernel` |
| 13 | Attention | **Cascade Multi-State** | [`flashinfer/.../cascade.py:99`](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/triton/kernels/cascade.py#L99) `merge_states_kernel` |
| 14 | Attention | **Merge Attention States** | [`vllm/.../triton_merge_attn_states.py:44`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_merge_attn_states.py#L44) `merge_attn_states_kernel` |
| 15 | Attention | **SGLang Extend** | [`sglang/.../extend_attention.py:219`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/triton_ops/extend_attention.py#L219) `_fwd_kernel` |
| 16 | Linear | **KDA Fused Recurrent** | [`fla/ops/kda/fused_recurrent.py:25`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/fused_recurrent.py#L25) `fused_recurrent_kda_fwd_kernel` |
| 17 | Linear | **KDA Chunk Intra** | [`fla/ops/kda/chunk_intra.py:36`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/chunk_intra.py#L36) `chunk_kda_fwd_kernel_inter_solve_fused` |
| 18 | Linear | **Gated DeltaNet Chunk FWD** | [`GatedDeltaNet/.../chunk.py:102`](https://github.com/NVlabs/GatedDeltaNet/blob/main/lit_gpt/gated_delta_rule_ops/chunk.py#L102) `chunk_gated_delta_rule_fwd_kernel_h` |
| 19 | Linear | **Gated DeltaNet Chunk BWD** | [`GatedDeltaNet/.../chunk.py:251`](https://github.com/NVlabs/GatedDeltaNet/blob/main/lit_gpt/gated_delta_rule_ops/chunk.py#L251) `chunk_gated_delta_rule_bwd_kernel_dhu` |
| 20 | Linear | **FLA Gated DeltaNet Fused** | [`fla/ops/gated_delta_rule/fused_recurrent.py:20`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/fused_recurrent.py#L20) `fused_recurrent_gated_delta_rule_fwd_kernel` |
| 21 | Linear | **GLA Chunk Output** | [`fla/ops/gla/chunk.py:28`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gla/chunk.py#L28) `chunk_gla_fwd_kernel_o` |
| 22 | Linear | **GLA Fused Recurrent** | [`fla/ops/gla/fused_recurrent.py:20`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gla/fused_recurrent.py#L20) `fused_recurrent_gla_fwd_kernel` |
| 23 | Linear | **Common Chunk Output** | [`fla/ops/common/chunk_o.py:30`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/common/chunk_o.py#L30) `chunk_fwd_kernel_o` |
| 24 | Linear | **RWKV-6 Fused Recurrent** | [`fla/ops/rwkv6/fused_recurrent.py:26`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/rwkv6/fused_recurrent.py#L26) `fused_recurrent_rwkv6_fwd_kernel` |
| 25 | Linear | **RWKV-7 Fused Recurrent** | [`fla/ops/rwkv7/fused_recurrent.py:30`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/rwkv7/fused_recurrent.py#L30) `fused_recurrent_rwkv7_fwd_kernel` |
| 26 | Linear | **Mamba-2 Chunk Scan** | [`mamba_ssm/.../ssd_chunk_scan.py:48`](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/ssd_chunk_scan.py#L48) `_chunk_scan_fwd_kernel` |
| 27 | Linear | **Mamba Selective Scan** | [`mamba_ssm/.../selective_state_update.py:23`](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/selective_state_update.py#L23) `_selective_scan_update_kernel` |
| 28 | Linear | **Mamba State Passing** | [`mamba_ssm/.../ssd_state_passing.py:29`](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/ssd_state_passing.py#L29) `_state_passing_fwd_kernel` |
| 29 | Sparse | **NSA TopK Attention** | [`native_sparse.../topk_sparse_attention.py:26`](https://github.com/XunhaoLai/native-sparse-attention-triton/blob/main/native_sparse_attention/ops/triton/topk_sparse_attention.py#L26) `forward_kernel` |
| 30 | Sparse | **NSA Compressed** | [`native_sparse.../compressed_attention.py:27`](https://github.com/XunhaoLai/native-sparse-attention-triton/blob/main/native_sparse_attention/ops/triton/compressed_attention.py#L27) `forward_kernel` |
| 31 | Sparse | **NSA Parallel (fla-org)** | [`native-sparse.../parallel.py:471`](https://github.com/fla-org/native-sparse-attention/blob/main/native_sparse_attention/ops/parallel.py#L471) `parallel_nsa_fwd_kernel` |
| 32 | Sparse | **Double Sparsity Attention** | [`sglang/.../double_sparsity_attention.py:22`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/triton_ops/double_sparsity_attention.py#L22) `_sparse_fwd_kernel_stage1` |
| 33 | Quantized | **SageAttention INT8** | [`sageattention/.../attn_qk_int8_per_block.py:75`](https://github.com/thu-ml/SageAttention/blob/main/sageattention/triton/attn_qk_int8_per_block.py#L75) `_attn_fwd` |
| 34 | Quantized | **SageAttention INT8 Causal** | [`sageattention/.../attn_qk_int8_per_block_causal.py:68`](https://github.com/thu-ml/SageAttention/blob/main/sageattention/triton/attn_qk_int8_per_block_causal.py#L68) `_attn_fwd` |
| 35 | MoE | **Fused MoE** | [`vllm/.../fused_moe.py:314`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py#L314) `fused_moe_kernel` |
| 36 | MoE | **Fused MoE GPTQ/AWQ** | [`vllm/.../fused_moe.py:81`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py#L81) `fused_moe_kernel_gptq_awq` |
| 37 | MoE | **SGLang Fused MoE** | [`sglang/.../fused_moe_triton_kernels.py:334`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py#L334) `fused_moe_kernel` |
| 38 | MoE | **EP MoE Permute** | [`sglang/.../ep_moe/kernels.py:73`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/ep_moe/kernels.py#L73) `deepep_permute_triton_kernel` |
| 39 | MoE | **MoE Router Tensor Core** | [`sglang/.../router.py:159`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/router.py#L159) `fused_moe_router_tensorcore_kernel` |
| 40 | Quant | **FP8 Block Matmul** | [`vllm/.../fp8_kernel.py:28`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/kernels/triton/fp8_kernel.py#L28) `_w8a8_block_fp8_matmul` |
| 41 | Quant | **INT8 Block Matmul** | [`vllm/.../int8_kernel.py:20`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/kernels/triton/int8_kernel.py#L20) `_w8a8_block_int8_matmul` |
| 42 | Quant | **AWQ GEMM** | [`vllm/.../awq_triton.py:76`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/kernels/triton/awq_triton.py#L76) `awq_gemm_kernel` |
| 43 | Quant | **Scaled MM** | [`vllm/.../triton_scaled_mm.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/kernels/triton/triton_scaled_mm.py) `scaled_mm_kernel` |
| 44 | Sampling | **Top-K Log Softmax** | [`vllm/.../triton_sampler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/triton_sampler.py) `_topk_log_softmax_kernel` |
| 45 | Sampling | **Temperature Scaling** | [`vllm/.../triton_sampler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/triton_sampler.py) `_temperature_kernel` |
| 46 | Sampling | **Min-P Filtering** | [`vllm/.../triton_sampler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/triton_sampler.py) `_min_p_kernel` |
| 47 | Sampling | **Penalties (Rep/Freq)** | [`vllm/.../triton_sampler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/triton_sampler.py) `_penalties_kernel`, `_bincount_kernel` |
| 48 | Sampling | **Gumbel Sampling** | [`vllm/.../triton_sampler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/triton_sampler.py) `_gumbel_sample_kernel` |
| 49 | Sampling | **EAGLE Prepare Inputs** | [`vllm/.../spec_decode/utils.py:6`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/utils.py#L6) `eagle_prepare_inputs_padded_kernel` |
| 50 | Sampling | **Rejection Sampling** | [`vllm/.../rejection_sample.py:9`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/spec_decode/rejection_sample.py#L9) `_rejection_sample_kernel` |
| 51 | LoRA | **LoRA Expand** | [`vllm/.../lora_expand.py:21`](https://github.com/vllm-project/vllm/blob/main/vllm/lora/ops/triton_ops/lora_expand.py#L21) `_lora_expand_kernel` |
| 52 | LoRA | **LoRA Shrink** | [`vllm/.../lora_shrink.py:21`](https://github.com/vllm-project/vllm/blob/main/vllm/lora/ops/triton_ops/lora_shrink.py#L21) `_lora_shrink_kernel` |
| 53 | Norm | **RMSNorm + Static FP8 Quant** | [`vllm/.../fusion.py#L92`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/fusion.py#L92) `rms_norm_static_fp8_quant` (fusion pattern) |
| 54 | Norm | **Fused Add + RMSNorm + Static FP8** | [`vllm/.../fusion.py#L95`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/fusion.py#L95) `fused_add_rms_norm_static_fp8_quant` |
| 55 | Norm | **RMSNorm + Dynamic Per-Token Quant** | [`vllm/.../fusion.py#L96`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/fusion.py#L96) `rms_norm_dynamic_per_token_quant` |
| 56 | Norm | **RMSNorm + Per-Block Quant** | [`vllm/.../fusion.py#L102`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/fusion.py#L102) `rms_norm_per_block_quant` |
| 57 | Activation | **SiLU+Mul (SwiGLU)** | [`vllm/.../activation.py#L65`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/activation.py#L65) `silu_and_mul` (PyTorch op, Helion target) |
| 58 | Activation | **SiLU+Mul+FP8 Quant** | [`vllm/.../fp8_kernel.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/kernels/triton/fp8_kernel.py) `_silu_mul_per_token_group_quant_fp8_colmajor` |

---

## Attention Kernels by Family

Attention kernels organized by architecture family, with complexity indicators.

### GQA (Grouped Query Attention)
Production-critical standard attention for Llama, Mistral, Qwen:

| # | Kernel | LoC | Key Features | Code Pointer |
|---|--------|-----|--------------|--------------|
| 1 | **GQA Prefill** | ~254 | Standard FlashAttention with GQA and sliding window | [`vllm/.../triton_prefill_attention.py:36`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_prefill_attention.py#L36) |
| 2 | **GQA Decode Stage1** | ~240 | Two-stage FlashDecoding, partial softmax | [`vllm/.../triton_decode_attention.py:58`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_decode_attention.py#L58) |
| 3 | **GQA Decode Stage2** | ~240 | Reduction stage for paged KV cache | [`vllm/.../triton_decode_attention.py:494`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_decode_attention.py#L494) |
| 4 | **Grouped Decode** | ~200 | Optimized decode with head grouping | [`vllm/.../triton_decode_attention.py:248`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_decode_attention.py#L248) |

### MLA (Multi-head Latent Attention)
DeepSeek's attention with 28x KV cache compression:

| # | Kernel | LoC | Key Features | Code Pointer |
|---|--------|-----|--------------|--------------|
| 5 | **MLA Attention** | ~200 | Multi-head latent attention, KV compression | [`FlashMLA/benchmark/bench_flash_mla.py:135`](https://github.com/deepseek-ai/FlashMLA/blob/main/benchmark/bench_flash_mla.py#L135) |
| 6 | **MLA Softmax ReduceV** | ~150 | Specialized reduction for MLA decode | [`FlashMLA/benchmark/bench_flash_mla.py:273`](https://github.com/deepseek-ai/FlashMLA/blob/main/benchmark/bench_flash_mla.py#L273) |

### SWA (Sliding Window Attention)
Efficient local attention for Gemma 3 (5:1 compute reduction):

| # | Kernel | LoC | Key Features | Code Pointer |
|---|--------|-----|--------------|--------------|
| 7 | **SWA Forward** | ~200 | `_attn_fwd` + `_attn_fwd_inner`, sliding window masking | [`flash-attn2/src/.../forward.py:21`](https://github.com/MaxLSB/flash-attn2/blob/main/src/flash_attention/forward.py#L21) |
| 8 | **SWA Backward** | ~600 | `_attn_bwd_dk_dv`, `_attn_bwd_dq`, complex gradients | [`flash-attn2/src/.../backward.py:76`](https://github.com/MaxLSB/flash-attn2/blob/main/src/flash_attention/backward.py#L76) |

### Unified Attention
Combined prefill+decode kernels for hybrid batching:

| # | Kernel | LoC | Key Features | Code Pointer |
|---|--------|-----|--------------|--------------|
| 9 | **Unified 2D** | ~150 | Combined prefill+decode, dynamic dispatch | [`vllm/.../triton_unified_attention.py:56`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_unified_attention.py#L56) |
| 10 | **Unified 3D (Segmented)** | ~150 | Memory-efficient for long sequences | [`vllm/.../triton_unified_attention.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_unified_attention.py) |
| 11 | **Chunked Prefill Paged Decode** | ~280 | Hybrid batching, page table indirection | [`vllm/.../chunked_prefill_paged_decode.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/chunked_prefill_paged_decode.py) |

### Cascade / State Merge
Shared-prefix optimization for multi-turn inference:

| # | Kernel | LoC | Key Features | Code Pointer |
|---|--------|-----|--------------|--------------|
| 12 | **Cascade State Merge** | ~35 | Simple log-sum-exp merging | [`flashinfer/.../cascade.py:24`](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/triton/kernels/cascade.py#L24) |
| 13 | **Cascade Multi-State** | ~35 | Multi-state variant of merge | [`flashinfer/.../cascade.py:99`](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/triton/kernels/cascade.py#L99) |
| 14 | **Merge Attention States** | ~45 | vLLM's attention state merging | [`vllm/.../triton_merge_attn_states.py:44`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_merge_attn_states.py#L44) |

### SGLang Attention
SGLang's extend/prefill implementation:

| # | Kernel | LoC | Key Features | Code Pointer |
|---|--------|-----|--------------|--------------|
| 15 | **SGLang Extend** | ~250 | Extend/prefill kernel with KV cache | [`sglang/.../extend_attention.py:219`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/triton_ops/extend_attention.py#L219) |

### KDA (Kimi Delta Attention)
Moonshot AI's linear attention with 75% KV cache reduction:

| # | Kernel | LoC | Key Features | Code Pointer |
|---|--------|-----|--------------|--------------|
| 16 | **KDA Fused Recurrent** | ~250 | Fused recurrence for delta attention, fwd + bwd | [`fla/ops/kda/fused_recurrent.py:25`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/fused_recurrent.py#L25) |
| 17 | **KDA Chunk Intra** | ~1100 | Triangular matrix solve, bidirectional attention | [`fla/ops/kda/chunk_intra.py:36`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/chunk_intra.py#L36) |

### Gated DeltaNet
NVIDIA's ICLR 2025 linear attention with gating:

| # | Kernel | LoC | Key Features | Code Pointer |
|---|--------|-----|--------------|--------------|
| 18 | **Gated DeltaNet Chunk FWD** | ~290 | 4-way unrolled K≤256, matrix inversion | [`GatedDeltaNet/.../chunk.py:102`](https://github.com/NVlabs/GatedDeltaNet/blob/main/lit_gpt/gated_delta_rule_ops/chunk.py#L102) |
| 19 | **Gated DeltaNet Chunk BWD** | ~285 | Complex gradient computation | [`GatedDeltaNet/.../chunk.py:251`](https://github.com/NVlabs/GatedDeltaNet/blob/main/lit_gpt/gated_delta_rule_ops/chunk.py#L251) |
| 20 | **FLA Gated DeltaNet Fused** | ~400 | Community-optimized, fwd + bwd | [`fla/ops/gated_delta_rule/fused_recurrent.py:20`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/fused_recurrent.py#L20) |

### GLA (Gated Linear Attention)
Foundation for KDA and Gated DeltaNet:

| # | Kernel | LoC | Key Features | Code Pointer |
|---|--------|-----|--------------|--------------|
| 21 | **GLA Chunk Output** | ~1350 | 4 fwd + 5 bwd kernels, extensive autotuning | [`fla/ops/gla/chunk.py:28`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gla/chunk.py#L28) |
| 22 | **GLA Fused Recurrent** | ~250 | Gated linear attention recurrence | [`fla/ops/gla/fused_recurrent.py:20`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gla/fused_recurrent.py#L20) |
| 23 | **Common Chunk Output** | ~690 | Shared by all FLA variants, chunk gating | [`fla/ops/common/chunk_o.py:30`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/common/chunk_o.py#L30) |

### RWKV
Receptance Weighted Key Value architectures:

| # | Kernel | LoC | Key Features | Code Pointer |
|---|--------|-----|--------------|--------------|
| 24 | **RWKV-6 Fused Recurrent** | ~350 | Fused recurrence, time-mixing, fwd + bwd | [`fla/ops/rwkv6/fused_recurrent.py:26`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/rwkv6/fused_recurrent.py#L26) |
| 25 | **RWKV-7 Fused Recurrent** | ~334 | Latest RWKV with 1.37x FlashAttn speedup | [`fla/ops/rwkv7/fused_recurrent.py:30`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/rwkv7/fused_recurrent.py#L30) |

### Mamba
State-space models with selective scan:

| # | Kernel | LoC | Key Features | Code Pointer |
|---|--------|-----|--------------|--------------|
| 26 | **Mamba-2 Chunk Scan** | ~350 | State-space chunk scanning, recurrence unrolling | [`mamba_ssm/.../ssd_chunk_scan.py:48`](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/ssd_chunk_scan.py#L48) |
| 27 | **Mamba Selective Scan** | ~200 | Selective state update, input-dependent dynamics | [`mamba_ssm/.../selective_state_update.py:23`](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/selective_state_update.py#L23) |
| 28 | **Mamba State Passing** | ~150 | Inter-chunk state propagation | [`mamba_ssm/.../ssd_state_passing.py:29`](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/ssd_state_passing.py#L29) |

### NSA (Native Sparse Attention)
Hardware-aligned sparse attention (29ms vs 810ms FlashAttn):

| # | Kernel | LoC | Key Features | Code Pointer |
|---|--------|-----|--------------|--------------|
| 29 | **NSA TopK Attention** | ~350 | Sparse index gathering, topk selection | [`native_sparse.../topk_sparse_attention.py:26`](https://github.com/XunhaoLai/native-sparse-attention-triton/blob/main/native_sparse_attention/ops/triton/topk_sparse_attention.py#L26) |
| 30 | **NSA Compressed** | ~300 | Compression ratios, multi-scale attention | [`native_sparse.../compressed_attention.py:27`](https://github.com/XunhaoLai/native-sparse-attention-triton/blob/main/native_sparse_attention/ops/triton/compressed_attention.py#L27) |
| 31 | **NSA Parallel (fla-org)** | ~1435 | Full fwd/bwd, 11+ kernels, complex state management | [`native-sparse.../parallel.py:471`](https://github.com/fla-org/native-sparse-attention/blob/main/native_sparse_attention/ops/parallel.py#L471) |

### Double Sparsity
SGLang's block-sparse implementation:

| # | Kernel | LoC | Key Features | Code Pointer |
|---|--------|-----|--------------|--------------|
| 32 | **Double Sparsity Attention** | ~300 | Block-sparse patterns, two-stage sparsity | [`sglang/.../double_sparsity_attention.py:22`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/triton_ops/double_sparsity_attention.py#L22) |

### SageAttention (Quantized Attention)
INT8 attention with 2-5x speedup vs FlashAttention:

| # | Kernel | LoC | Key Features | Code Pointer |
|---|--------|-----|--------------|--------------|
| 33 | **SageAttention INT8** | ~280 | INT8 quantization per block, mixed precision | [`sageattention/.../attn_qk_int8_per_block.py:75`](https://github.com/thu-ml/SageAttention/blob/main/sageattention/triton/attn_qk_int8_per_block.py#L75) |
| 34 | **SageAttention INT8 Causal** | ~280 | Causal variant for autoregressive | [`sageattention/.../attn_qk_int8_per_block_causal.py:68`](https://github.com/thu-ml/SageAttention/blob/main/sageattention/triton/attn_qk_int8_per_block_causal.py#L68) |

---

## Non-Attention Kernel Complexity Ranking

Based on actual source code analysis of lines of code, number of kernels per file, algorithmic complexity, autotune configurations, and forward/backward pass implementations.

### Tier 1: Complex (300+ lines, specialized techniques)

| Rank | Kernel | LoC | Kernels | Key Complexity Factors |
|------|--------|-----|---------|----------------------|
| 1 | **Fused MoE** #35 | ~500 | 1 | Block-sparse matmul, token routing, FP8/INT8 quantization, channel-wise scaling, persistent kernel |
| 2 | **Fused MoE GPTQ/AWQ** #36 | ~230 | 1 | 4-bit weight dequantization, mixed-precision accumulation, multiple pack formats |
| 3 | **SGLang Fused MoE** #37 | ~250 | 1 | Similar to vLLM fused MoE, token routing |

### Tier 2: Moderate (150-300 lines, standard patterns)

| Rank | Kernel | LoC | Kernels | Key Complexity Factors |
|------|--------|-----|---------|----------------------|
| 4 | **FP8 Block Matmul** #40 | ~180 | 1 | Block-wise FP8 scaling, fused dequantization |
| 5 | **AWQ GEMM** #42 | ~200 | 1 | 4-bit weight unpacking, per-group scales |
| 6 | **EP MoE Permute** #38 | ~180 | 1 | Expert-parallel token permutation |
| 7 | **MoE Router Tensor Core** #39 | ~160 | 1 | Tensor-core optimized routing computation |
| 8 | **INT8 Block Matmul** #41 | ~150 | 1 | Block-wise INT8 scaling |
| 9 | **Top-K Log Softmax** #44 | ~120 | 1 | Parallel top-k with log softmax |

### Tier 3: Simpler (Under 150 lines, straightforward logic)

| Rank | Kernel | LoC | Kernels | Key Complexity Factors |
|------|--------|-----|---------|----------------------|
| 10 | **RMSNorm + Static FP8** #53 | ~100 | 1 | Fused norm + quantization |
| 11 | **Fused Add + RMSNorm + FP8** #54 | ~110 | 1 | Residual + norm + quant fusion |
| 12 | **RMSNorm + Dynamic Quant** #55 | ~110 | 1 | Per-token dynamic scaling |
| 13 | **RMSNorm + Per-Block Quant** #56 | ~110 | 1 | Per-block scaling variant |
| 14 | **Scaled MM** #43 | ~100 | 1 | Generic scaled matmul |
| 15 | **SiLU+Mul+FP8 Quant** #58 | ~90 | 1 | Fused activation + quantization |
| 16 | **LoRA Expand** #51 | ~80 | 1 | Simple matmul with LoRA indices |
| 17 | **LoRA Shrink** #52 | ~80 | 1 | Transpose variant of expand |
| 18 | **EAGLE Prepare Inputs** #49 | ~80 | 1 | Tree structure preparation |
| 19 | **Penalties (Rep/Freq)** #47 | ~80 | 2 | Penalty application + bincount |
| 20 | **Rejection Sampling** #50 | ~70 | 1 | Draft token verification |
| 21 | **Min-P Filtering** #46 | ~60 | 1 | Threshold filtering |
| 22 | **Gumbel Sampling** #48 | ~50 | 1 | Gumbel-max trick |
| 23 | **SiLU+Mul (SwiGLU)** #57 | ~50 | 1 | Element-wise activation |
| 24 | **Temperature Scaling** #45 | ~40 | 1 | Simple element-wise scaling |

### Complexity Scoring Methodology

**Factors considered (weighted):**
1. **Lines of Code (25%)**: Raw kernel size excluding comments/whitespace
2. **Number of Kernels (20%)**: Separate `@triton.jit` functions in the file
3. **Algorithmic Complexity (30%)**: Matrix operations, quantization logic, routing
4. **Autotune Configurations (10%)**: Number of `@triton.autotune` configs
5. **Memory Access Patterns (15%)**: Indirect indexing, token permutation

---

## Detailed Breakdown by Category (Non-Attention)

### MoE (5 kernels)
Mixture of Experts routing and computation:

35. **Fused MoE** - Production FP8 MoE for DeepSeek-V3, Llama-4
36. **Fused MoE GPTQ/AWQ** - Quantized MoE variant
37. **SGLang Fused MoE** - SGLang's MoE implementation
38. **EP MoE Permute** - Expert parallel token permutation
39. **MoE Router Tensor Core** - Tensor-core optimized routing

### Quantization (4 kernels)
Block-wise quantization for memory efficiency:

40. **FP8 Block Matmul** - DeepSeek-V3, Llama-4 inference
41. **INT8 Block Matmul** - W8A8 block-scaled GEMM
42. **AWQ GEMM** - 4-bit weight quantization
43. **Scaled MM** - Generic scaled matrix multiply

### Sampling (7 kernels)
Efficient token selection (see [FlashInfer Sampling Blog](https://flashinfer.ai/2025/03/10/sampling.html)):

44. **Top-K Log Softmax** - Fused top-k + log softmax (10x speedup vs sorting)
45. **Temperature Scaling** - Logit temperature adjustment before sampling
46. **Min-P Filtering** - Filter tokens below `p_base × p_max` threshold
47. **Penalties (Rep/Freq)** - Repetition and frequency penalty application with bincount
48. **Gumbel Sampling** - Gumbel-max trick for efficient categorical sampling
49. **EAGLE Prepare Inputs** - Speculative decoding tree preparation
50. **Rejection Sampling** - Draft token verification for speculative decoding

### LoRA (2 kernels)
Low-rank adaptation:

51. **LoRA Expand** - Latent to hidden mapping
52. **LoRA Shrink** - Hidden to latent mapping

### Normalization/Activation (6 kernels)
Fundamental building blocks (see [vLLM #32962](https://github.com/vllm-project/vllm/issues/32962)):

53. **RMSNorm + Static FP8 Quant** - Fused norm with static FP8 quantization
54. **Fused Add + RMSNorm + Static FP8** - Residual connection + norm + quant
55. **RMSNorm + Dynamic Per-Token Quant** - Fused norm with dynamic per-token FP8
56. **RMSNorm + Per-Block Quant** - Fused norm with per-block FP8 quantization
57. **SiLU+Mul (SwiGLU)** - Standalone SwiGLU activation
58. **SiLU+Mul+FP8 Quant** - Fused SwiGLU activation with quantization

---

### Non-Attention Kernels by Priority

**Tier 1: Critical Path**
- Fused MoE (35)
- FP8 Block Matmul (40)
- Sampling Suite: Top-K, Temperature, Min-P, Gumbel (44-48)
- LoRA Expand/Shrink (51-52)

**Tier 2: Optimization**
- EAGLE/Rejection Sampling (49-50)
- Fused MoE GPTQ/AWQ (36)
- SGLang Fused MoE (37)

**Tier 3: Foundation**
- Quantization kernels (41-43)
- Fused normalization suite: RMSNorm + quantization variants (53-56)
- Activation suite: SiLU+Mul standalone and fused variants (57-58)
- MoE utilities (38-39)

---

## Source Repositories

| Repository | URL | Kernel Count |
|------------|-----|--------------|
| vLLM | https://github.com/vllm-project/vllm | 30 |
| SGLang | https://github.com/sgl-project/sglang | 10 |
| flash-linear-attention | https://github.com/fla-org/flash-linear-attention | 8 |
| mamba-ssm | https://github.com/state-spaces/mamba | 4 |
| GatedDeltaNet | https://github.com/NVlabs/GatedDeltaNet | 2 |
| native-sparse-attention-triton | https://github.com/XunhaoLai/native-sparse-attention-triton | 2 |
| FlashMLA | https://github.com/deepseek-ai/FlashMLA | 2 |
| SageAttention | https://github.com/thu-ml/SageAttention | 2 |
| FlashInfer | https://github.com/flashinfer-ai/flashinfer | 2 |
| flash-attn2 | https://github.com/MaxLSB/flash-attn2 | 2 |

---

## References

- [FlashInfer Sampling Blog](https://flashinfer.ai/2025/03/10/sampling.html) - Efficient GPU sampling algorithms
- [FlashInfer Cascade Inference](https://flashinfer.ai/2024/02/02/cascade-inference.html) - Shared-prefix optimization
- [vLLM Triton Sampler PR #25824](https://github.com/vllm-project/vllm/pull/25824) - 10x sampling speedup
- [vLLM Custom Helion Kernels #32962](https://github.com/vllm-project/vllm/issues/32962) - Target fusion patterns for Helion
