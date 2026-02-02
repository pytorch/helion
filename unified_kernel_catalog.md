# Unified GPU Kernel Catalog for LLM Inference, for Helion Hackathon

A deduplicated reference combining attention mechanisms, linear attention variants, and specialized inference kernels with their latest model usage and Triton implementations.

---

## Quick Reference

| Category | Kernel | Latest Model (2025) | Key Triton Code Pointer |
|----------|--------|---------------------|------------------------|
| **Attention** | GQA | Mistral Large 3 (Dec 2025) | [`vllm/.../triton_prefill_attention.py:36`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_prefill_attention.py#L36) `_fwd_kernel` |
| **Attention** | MLA | Kimi K2 (Jul 2025) | [`FlashMLA/benchmark/bench_flash_mla.py:135`](https://github.com/deepseek-ai/FlashMLA/blob/main/benchmark/bench_flash_mla.py#L135) `_mla_attn_kernel` |
| **Attention** | MLA Softmax ReduceV | MLA decode | [`FlashMLA/benchmark/bench_flash_mla.py:273`](https://github.com/deepseek-ai/FlashMLA/blob/main/benchmark/bench_flash_mla.py#L273) `_mla_softmax_reducev_kernel` |
| **Attention** | SWA | Gemma 3 (Mar 2025) | [`flash-attn2/src/.../forward.py:21`](https://github.com/MaxLSB/flash-attn2/blob/main/src/flash_attention/forward.py#L21) `_attn_fwd`, `_attn_fwd_inner` (MODE==2) |
| **Attention** | SWA Backward | SWA training kernels | [`flash-attn2/src/.../backward.py:76`](https://github.com/MaxLSB/flash-attn2/blob/main/src/flash_attention/backward.py#L76) `_attn_bwd_dk_dv`, `_attn_bwd_dq` |
| **Attention** | KDA | Kimi Linear (Oct 2025) | [`fla/ops/kda/fused_recurrent.py:25`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/fused_recurrent.py#L25) `fused_recurrent_kda_fwd_kernel` |
| **Attention** | KDA Chunk | KDA chunk-wise kernels | [`fla/ops/kda/chunk_intra.py:36`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/chunk_intra.py#L36) `chunk_kda_fwd_kernel_inter_solve_fused`, `chunk_kda_bwd_kernel_intra` |
| **Attention** | KDA Gate | KDA gate computation | [`fla/ops/kda/gate.py:81`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/gate.py#L81) `kda_gate_fwd_kernel`, `kda_gate_bwd_kernel` |
| **Attention** | SGLang KDA | KDA integration (7 kernels) | [`sglang/.../fla/kda.py:158`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/fla/kda.py#L158) |
| **Attention** | Paged (Prefill/Decode) | All decoder LLMs | [`vllm/.../triton_decode_attention.py:58`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_decode_attention.py#L58) `_fwd_kernel_stage1`, `:248` `_fwd_grouped_kernel_stage1` |
| **Attention** | Paged Decode Stage 2 | Decode reduction stage | [`vllm/.../triton_decode_attention.py:494`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_decode_attention.py#L494) `_fwd_kernel_stage2` |
| **Attention** | SGLang Prefill | All decoder LLMs | [`sglang/.../prefill_attention.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/triton_ops/prefill_attention.py) `_fwd_kernel` |
| **Attention** | Unified 3D (Segmented) | vLLM long sequences | [`vllm/.../triton_unified_attention.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_unified_attention.py) `kernel_unified_attention_3d` |
| **Attention** | Chunked Prefill Paged Decode | Hybrid batching | [`vllm/.../chunked_prefill_paged_decode.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/chunked_prefill_paged_decode.py) `chunked_prefill_paged_decode_kernel` |
| **Attention** | Cascade | Shared-prefix batching | [`flashinfer/.../cascade.py:24`](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/triton/kernels/cascade.py#L24) `merge_state_kernel` |
| **Attention** | Cascade In-Place | In-place state merge | [`flashinfer/.../cascade.py:61`](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/triton/kernels/cascade.py#L61) `merge_state_in_place_kernel` |
| **Attention** | Cascade Multi-State | Multi-state merging | [`flashinfer/.../cascade.py:99`](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/triton/kernels/cascade.py#L99) `merge_states_kernel`, `variable_length_merge_states_kernel` |
| **Attention** | Merge Attention States | Attention state merging | [`vllm/.../triton_merge_attn_states.py:44`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_merge_attn_states.py#L44) `merge_attn_states_kernel` |
| **Attention** | Gated Attention | Qwen (NeurIPS 2025) | PyTorch-only (no Triton kernels) |
| **Linear** | Gated DeltaNet | Qwen3-Next (Sep 2025) | [`GatedDeltaNet/.../chunk.py:102`](https://github.com/NVlabs/GatedDeltaNet/blob/main/lit_gpt/gated_delta_rule_ops/chunk.py#L102) `chunk_gated_delta_rule_fwd_kernel_h`, `chunk_gated_delta_rule_bwd_kernel_dhu` |
| **Linear** | FLA Gated DeltaNet | FLA-compatible implementation | [`fla/ops/gated_delta_rule/fused_recurrent.py:20`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/fused_recurrent.py#L20), [`wy_fast.py:22`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/wy_fast.py#L22) |
| **Linear** | GatedDeltaNet (NVlabs FLA) | NVlabs FLA-compatible chunks | [`GatedDeltaNet/.../fla_version/chunk_fla.py:588`](https://github.com/NVlabs/GatedDeltaNet/blob/main/lit_gpt/gated_delta_rule_ops/fla_version/chunk_fla.py#L588) `chunk_gated_delta_rule_fwd_kernel_h` |
| **Linear** | GLA (Gated Linear Attention) | GLA-based models | [`fla/ops/gla/chunk.py:28`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gla/chunk.py#L28) `chunk_gla_fwd_kernel_o`, `chunk_gla_bwd_kernel_dv` |
| **Linear** | GLA Fused Recurrent | Fused recurrent GLA | [`fla/ops/gla/fused_recurrent.py:20`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gla/fused_recurrent.py#L20) `fused_recurrent_gla_fwd_kernel` |
| **Linear** | Common Chunk Output | All FLA variants | [`fla/ops/common/chunk_o.py:30`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/common/chunk_o.py#L30) `chunk_fwd_kernel_o` |
| **Linear** | Triangular Solve | All linear attention | [`fla/ops/utils/solve.py:23`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/solve.py#L23) `solve_tril_16x16_kernel` |
| **Linear** | RetNet/RWKV-6 | RWKV-6 models | [`fla/ops/rwkv6/fused_recurrent.py:26`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/rwkv6/fused_recurrent.py#L26) |
| **Linear** | RWKV-7 | RWKV-7 (2025) | [`fla/ops/rwkv7/fused_recurrent.py:30`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/rwkv7/fused_recurrent.py#L30), [`channel_mixing.py:33`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/rwkv7/channel_mixing.py#L33) |
| **Linear** | RWKV-4 | RWKV-4 models | [`fla/ops/rwkv4/fused_recurrent.py:23`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/rwkv4/fused_recurrent.py#L23) |
| **Linear** | Mamba-2 | Mamba-2 (2024) | [`mamba_ssm/.../ssd_chunk_scan.py:48`](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/ssd_chunk_scan.py#L48) `_chunk_scan_fwd_kernel` |
| **Sparse** | NSA | DeepSeek V3.2 (2025) | [`native_sparse.../topk_sparse_attention.py:26`](https://github.com/XunhaoLai/native-sparse-attention-triton/blob/main/native_sparse_attention/ops/triton/topk_sparse_attention.py#L26) `forward_kernel` |
| **Sparse** | NSA Compressed | Compressed attention branch | [`native_sparse.../compressed_attention.py:27`](https://github.com/XunhaoLai/native-sparse-attention-triton/blob/main/native_sparse_attention/ops/triton/compressed_attention.py#L27) `forward_kernel`, `score_kernel` |
| **Sparse** | NSA Parallel (fla-org) | Full training support | [`native-sparse.../parallel.py:471`](https://github.com/fla-org/native-sparse-attention/blob/main/native_sparse_attention/ops/parallel.py#L471) `parallel_nsa_fwd_kernel`, `parallel_nsa_kernel_topk` |
| **Sparse** | NSA Selection (nsa-impl) | Selection attention kernels | [`nsa/selection.py:35`](https://github.com/tilde-research/nsa-impl/blob/main/nsa/selection.py#L35) `_sel_attn_fwd_kernel`, `_sel_attn_bwd_kernel` |
| **Sparse** | SGLang NSA | NSA integration | [`sglang/.../nsa/triton_kernel.py:9`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/nsa/triton_kernel.py#L9) |
| **Sparse** | Block Sparse | FlashAttention 2.4+ | CUDA-only (no Triton); see [`SGLang double_sparsity`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/triton_ops/double_sparsity_attention.py#L22) |
| **Quantized** | SageAttention | General (FP4/FP8/INT8) | [`sageattention/.../attn_qk_int8_per_block.py:75`](https://github.com/thu-ml/SageAttention/blob/main/sageattention/triton/attn_qk_int8_per_block.py#L75) `_attn_fwd`, `_attn_fwd_inner` |
| **Quantized** | SageAttention INT8 Quant | Per-block INT8 quantization | [`sageattention/.../quant_per_block.py:21`](https://github.com/thu-ml/SageAttention/blob/main/sageattention/triton/quant_per_block.py#L21) `quant_per_block_int8_kernel` |
| **Quantized** | SageAttention INT8 Causal | Causal INT8 attention | [`sageattention/.../attn_qk_int8_per_block_causal.py:68`](https://github.com/thu-ml/SageAttention/blob/main/sageattention/triton/attn_qk_int8_per_block_causal.py#L68) `_attn_fwd` |
| **Quantized** | SageAttention INT4 Quant | INT4 query/key quantization | [`sageattention/.../quant_per_thread.py:100`](https://github.com/thu-ml/SageAttention/blob/main/sageattention/triton/quant_per_thread.py#L100) `quant_query_per_thread_int4_kernel` |
| **MoE** | Fused MoE + FP8 | DeepSeek-V3, Llama-4 | [`vllm/.../fused_moe.py:314`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py#L314) `fused_moe_kernel` |
| **MoE** | SGLang Fused MoE | Fused MoE with activations | [`sglang/.../fused_moe_triton_kernels.py:334`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py#L334) `fused_moe_kernel` |
| **MoE** | Fused MoE GPTQ/AWQ | Quantized MoE models | [`vllm/.../fused_moe.py:81`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py#L81) `fused_moe_kernel_gptq_awq` |
| **MoE** | MoE Align Block Size | Token-to-expert mapping | [`sglang/.../fused_moe_triton_kernels.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py) `moe_align_block_size_stage1/2/3/4` |
| **MoE** | Compute Identity | Expert identity computation | [`vllm/.../fused_moe.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py) `compute_identity_kernel` |
| **MoE** | DeepEP Post-Reorder | Output reordering for DeepEP | [`sglang/.../ep_moe/kernels.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/ep_moe/kernels.py) `deepep_post_reorder_triton_kernel` |
| **MoE** | EP Scatter/Gather | Expert parallelism distribution | [`vllm/.../fused_moe.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py) `_fwd_kernel_ep_scatter_1/2`, `_fwd_kernel_ep_gather` |
| **Sampling** | Top-K/Top-P/Min-P | All autoregressive LLMs | [vLLM PR #25824](https://github.com/vllm-project/vllm/pull/25824) (env: `VLLM_USE_TRITON_SAMPLER=1`) |
| **Sampling** | Top-K Log Softmax | Top-K with softmax | [`vllm/.../triton_sampler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/triton_sampler.py) `_topk_log_softmax_kernel` |
| **Sampling** | Temperature | Temperature scaling | [`vllm/.../triton_sampler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/triton_sampler.py) `_temperature_kernel` |
| **Sampling** | Min-P | Min-p probability filtering | [`vllm/.../triton_sampler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/triton_sampler.py) `_min_p_kernel` |
| **Sampling** | Penalties | Repetition/frequency penalties | [`vllm/.../triton_sampler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/triton_sampler.py) `_penalties_kernel`, `_bincount_kernel` |
| **Sampling** | Speculative Decoding | EAGLE-2, Medusa, SpecInfer | [`vllm/.../spec_decode/utils.py:6`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/utils.py#L6) `eagle_prepare_inputs_padded_kernel` |
| **Sampling** | EAGLE Next Token | EAGLE next token preparation | [`vllm/.../spec_decode/utils.py:49`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/utils.py#L49) `eagle_prepare_next_token_padded_kernel` |
| **Sampling** | Rejection Sampling | Speculative decoding verification | [`vllm/.../rejection_sample.py:9`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/spec_decode/rejection_sample.py#L9) `_rejection_sample_kernel` |
| **Sampling** | SGLang EAGLE Info V2 | EAGLE cache management | [`sglang/.../eagle_info_v2.py:53`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/speculative/eagle_info_v2.py#L53) `assign_draft_cache_locs_page_size_1` |
| **Sampling** | SGLang Spec Utils | Speculative decoding utilities | [`sglang/.../spec_utils.py:61`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/speculative/spec_utils.py#L61) `create_extend_after_decode_spec_info`, `assign_draft_cache_locs` |
| **Sampling** | SGLang Multi-Layer EAGLE | Multi-layer EAGLE utilities | [`sglang/.../multi_layer_eagle_utils.py:20`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/speculative/multi_layer_eagle_utils.py#L20) `rotate_input_ids_kernel` |
| **Quantization** | AWQ (W4A16) | Llama, Mistral, Qwen | [`vllm/.../awq_triton.py:31`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/kernels/triton/awq_triton.py#L31) `awq_dequantize_kernel` |
| **Quantization** | SGLang AWQ | Adapted from vLLM | [`sglang/.../awq_triton.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/awq_triton.py) |
| **Quantization** | INT8 Per-Token Quant | Token-wise INT8 quantization | [`vllm/.../int8_kernel.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/kernels/triton/int8_kernel.py) `_per_token_quant_int8` |
| **Quantization** | INT8 Per-Group Quant | DeepSeek-V3, Llama-4 | [`vllm/.../int8_utils.py:153`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/int8_utils.py#L153) `_per_token_group_quant_int8` |
| **Quantization** | INT8 Block Matmul | DeepSeek-V3, Llama-4 | [`vllm/.../int8_kernel.py:20`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/kernels/triton/int8_kernel.py#L20) `_w8a8_block_int8_matmul` |
| **Quantization** | SGLang INT8 Quant | Per-token/group INT8 | [`sglang/.../int8_kernel.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/int8_kernel.py) |
| **Quantization** | FP8 Per-Token Quant | Token-wise FP8 quantization | [`vllm/.../fp8_kernel.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/kernels/triton/fp8_kernel.py) `_per_token_group_quant_fp8` |
| **Quantization** | FP8 Block Matmul | DeepSeek-V3, Llama-4 | [`vllm/.../fp8_kernel.py:28`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/kernels/triton/fp8_kernel.py#L28) `_w8a8_block_fp8_matmul` |
| **Quantization** | SGLang FP8 Quant | Row/column-major scales | [`sglang/.../fp8_kernel.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/fp8_kernel.py) |
| **Quantization** | FP8 AMD Unrolled | 4x unrolled for AMD/HIP | [`sglang/.../fp8_kernel.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/fp8_kernel.py) `_w8a8_block_fp8_matmul_unrolledx4` |
| **Quantization** | Static FP8 | Per-tensor FP8 quant | [`sglang/.../fp8_kernel.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/fp8_kernel.py) `_static_quant_fp8` |
| **Quantization** | INT8 Column-Major | deep_gemm compatibility | [`sglang/.../int8_kernel.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/int8_kernel.py) `_per_token_group_quant_8bit_colmajor` |
| **Quantization** | Scale Swizzle | Scale layout optimization | [`vllm/.../triton_scaled_mm.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/kernels/triton/triton_scaled_mm.py) `triton_scale_swizzle` |
| **LoRA** | SGMV LoRA | All LoRA-fine-tuned models | [`vllm/.../lora_expand.py:21`](https://github.com/vllm-project/vllm/blob/main/vllm/lora/ops/triton_ops/lora_expand.py#L21) `_lora_expand_kernel` |
| **LoRA** | LoRA Core Helpers | Core expand/shrink operations | [`vllm/.../lora_expand.py`](https://github.com/vllm-project/vllm/blob/main/vllm/lora/ops/triton_ops/lora_expand.py) `do_expand_kernel`, [`lora_shrink.py`](https://github.com/vllm-project/vllm/blob/main/vllm/lora/ops/triton_ops/lora_shrink.py) `do_shrink_kernel` |
| **Activation** | SiLU + Mul (SwiGLU) | Llama, Mistral, DeepSeek | [`vllm/.../fused_moe.py:81`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py#L81) `fused_moe_kernel` (fused); [`sglang/.../fused_moe_triton_kernels.py:865`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py#L865) `act_and_mul_kernel` |
| **Activation** | SiLU+Mul+FP8 Quant | Fused activation with FP8 | [`vllm/.../fp8_kernel.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/kernels/triton/fp8_kernel.py) `_silu_mul_per_token_group_quant_fp8_colmajor` |
| **Normalization** | RMSNorm / LayerNorm | All transformer LLMs | [`mamba_ssm/.../layer_norm.py`](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/layer_norm.py) `_layer_norm_fwd_1pass_kernel` |
| **Normalization** | vLLM Mamba LayerNorm | Adapted Mamba layer norm | [`vllm/.../mamba/ops/layer_norm.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/ops/layer_norm.py) |
| **Sparse** | Double Sparsity Attention | Long-context inference | [`sglang/.../double_sparsity_attention.py:22`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/triton_ops/double_sparsity_attention.py#L22) `_sparse_fwd_kernel_stage1/2/3` |
| **Attention** | SGLang Decode | All decoder LLMs | [`sglang/.../decode_attention.py:44`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/triton_ops/decode_attention.py#L44) `_fwd_kernel_stage1`, `_fwd_grouped_kernel_stage1` |
| **Attention** | Unified 2D | vLLM combined prefill+decode | [`vllm/.../triton_unified_attention.py:56`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_unified_attention.py#L56) `kernel_unified_attention_2d` |
| **Attention** | SGLang Extend | Extend/prefill attention | [`sglang/.../extend_attention.py:219`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/triton_ops/extend_attention.py#L219) `_fwd_kernel`, `_fwd_kernel_unified` |
| **MoE** | EP MoE (Expert Parallel) | DeepSeek-V3, multi-GPU | [`sglang/.../ep_moe/kernels.py:73`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/ep_moe/kernels.py#L73) `deepep_permute_triton_kernel`, `_fwd_kernel_ep_gather` |
| **MoE** | MoE Router | Token-to-expert routing | [`sglang/.../router.py:13`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/router.py#L13) `fused_moe_router_cudacore_kernel`, `fused_moe_router_tensorcore_kernel` |
| **Linear** | Mamba Selective Scan | SSM state update | [`mamba_ssm/.../selective_state_update.py:23`](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/selective_state_update.py#L23) `_selective_scan_update_kernel` |
| **Linear** | Mamba Causal Conv1d | 1D causal convolution | [`mamba_ssm/.../causal_conv1d.py`](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/causal_conv1d.py) `_causal_conv1d_fwd_kernel` |
| **Attention** | FlashInfer Paged | Batch indices/positions | [`flashinfer/triton/page.py:21`](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/triton/page.py#L21) `get_batch_indices_positions_kernel` |
| **Quantization** | AWQ GEMM | Fused dequant + matmul | [`vllm/.../awq_triton.py:76`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/kernels/triton/awq_triton.py#L76) `awq_gemm_kernel` |
| **Quantization** | Scaled MM | Generic scaled matmul | [`vllm/.../triton_scaled_mm.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/kernels/triton/triton_scaled_mm.py) `scaled_mm_kernel` |
| **Quantization** | MxFP8 | Microscaling FP8 format | [`sglang/.../fp8_kernel.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/fp8_kernel.py) `_mxfp8_block_scaled_matmul_kernel` |
| **LoRA** | LoRA Shrink | Hidden → latent mapping | [`vllm/.../lora_shrink.py:21`](https://github.com/vllm-project/vllm/blob/main/vllm/lora/ops/triton_ops/lora_shrink.py#L21) `_lora_shrink_kernel` |
| **LoRA** | SGLang Chunked LoRA Expand | Segmented expand | [`sglang/.../lora_expand.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/triton_ops/lora_expand.py) `_chunked_lora_expand_kernel` |
| **LoRA** | SGLang Chunked LoRA Shrink | Segmented shrink | [`sglang/.../lora_shrink.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/triton_ops/lora_shrink.py) `_chunked_lora_shrink_kernel` |
| **LoRA** | SGLang QKV LoRA B | QKV-specific kernel | [`sglang/.../lora_ops.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/triton_ops/lora_ops.py) `_qkv_lora_b_kernel` |
| **LoRA** | SGLang Gate-Up LoRA B | MLP-specific kernel | [`sglang/.../lora_ops.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/triton_ops/lora_ops.py) `_gate_up_lora_b_kernel` |
| **LoRA** | SGLang SGEMM LoRA A/B | Layer-specific matmul | [`sglang/.../lora_ops.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/triton_ops/lora_ops.py) `_sgemm_lora_a_kernel`, `_sgemm_lora_b_kernel` |
| **Linear** | Mamba State Passing | SSM state propagation | [`mamba_ssm/.../ssd_state_passing.py:29`](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/ssd_state_passing.py#L29) `_state_passing_fwd_kernel` |
| **Attention** | SGLang MLA RoPE | ROCm MLA with RoPE | [`sglang/.../rocm_mla_decode_rope.py:44`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/triton_ops/rocm_mla_decode_rope.py#L44) `_fwd_grouped_kernel_stage1_rope` |
| **Linear** | Helion Gated DeltaNet | Helion DSL implementation | [`helion/examples/gdn_fwd_h.py:28`](https://github.com/pytorch-labs/helion/blob/main/examples/gdn_fwd_h.py#L28) `helion_gdn_fwd_h` |
| **Linear** | Mamba Chunk State | SSM chunk state | [`mamba_ssm/.../ssd_chunk_state.py:191`](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/ssd_chunk_state.py#L191) `_chunk_state_fwd_kernel` |
| **Linear** | vLLM Mamba ops | Adapted Mamba kernels | [`vllm/.../mamba/ops/`](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/mamba/ops) (7 kernel files) |
| **Linear** | SGLang Mamba ops | Adapted from vLLM | [`sglang/.../attention/mamba/`](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/layers/attention/mamba) |
| **Sampling** | Gumbel Sampling | Gumbel-max trick | [`vllm/.../triton_sampler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/triton_sampler.py) `_gumbel_sample_kernel` |
| **Linear** | vLLM FLA Scaled Dot | FLA in vLLM | [`vllm/.../chunk_scaled_dot.py`](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/fla/ops) `chunk_scaled_dot_kkt_fwd_kernel` |

---

## 1. Standard Attention Mechanisms

### GQA (Grouped-Query Attention)

**Concept**: Multiple query heads share fewer key-value heads, reducing KV cache memory while maintaining most of MHA's expressiveness.

| Latest Model | Release Date | Details |
|--------------|--------------|---------|
| **Mistral Large 3** | Dec 2, 2025 | GQA + MoE, 675B total/41B active, 128K context |
| Llama 4 | Apr 5, 2025 | GQA + MoE, 10M context |
| Qwen3 | Apr 28, 2025 | GQA + QK-Norm, 128K context |
| Gemma 3 | Mar 12, 2025 | GQA + SWA 5:1, 128K context |

**Triton Implementations**:
| Repository | Description | Code Pointer |
|------------|-------------|--------------|
| **vLLM Triton Prefill Attention** | Production-ready GQA kernel with query packing | [`vllm/v1/attention/ops/triton_prefill_attention.py:36`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_prefill_attention.py#L36) `_fwd_kernel` |
| **vLLM Triton Decode Attention** | GQA decode stage kernels | [`vllm/v1/attention/ops/triton_decode_attention.py:58`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_decode_attention.py#L58) `_fwd_kernel_stage1`, `:248` `_fwd_grouped_kernel_stage1` |
| **SGLang Decode Attention** | GQA decode with grouping support | [`sglang/srt/layers/attention/triton_ops/decode_attention.py:44`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/triton_ops/decode_attention.py#L44) `_fwd_kernel_stage1`, `:252` `_fwd_grouped_kernel_stage1` |

---

### MLA (Multi-Head Latent Attention)

**Concept**: Compresses K and V tensors into a low-dimensional latent space before caching. Achieves ~28x KV cache compression while maintaining MHA-level expressiveness.

| Latest Model | Release Date | Details |
|--------------|--------------|---------|
| **Kimi K2** | Jul 2025 | MLA, 1.04T total/32B active |
| DeepSeek R1 | Jan 20, 2025 | MLA + MoE, 671B total/37B active |

**Technical Notes**:
- KV cache reduced from ~213 GB to ~7.6 GB (128K context)
- Joint KV compression: Single down-projection matrix for both K and V
- RoPE handled via decoupled embeddings (separate rope_k dimension)

**Triton Implementations**:
| Repository | Description | Code Pointer |
|------------|-------------|--------------|
| **FlashMLA (Official)** | DeepSeek's official MLA kernels, 660 TFlops on H800 | [`benchmark/bench_flash_mla.py:135`](https://github.com/deepseek-ai/FlashMLA/blob/main/benchmark/bench_flash_mla.py#L135) `_mla_attn_kernel`, `:273` `_mla_softmax_reducev_kernel` |
| **SGLang MLA Backend** | MLA with RoPE support | [`sglang/srt/layers/attention/triton_ops/rocm_mla_decode_rope.py:44`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/triton_ops/rocm_mla_decode_rope.py#L44) `_fwd_grouped_kernel_stage1_rope` |
| **TransMLA** | 10.6x inference speedup variant | [arxiv.org/abs/2502.07864](https://arxiv.org/abs/2502.07864) |

---

### SWA (Sliding Window Attention)

**Concept**: Each token only attends to a fixed window of preceding tokens. Information flows through stacked layers to access longer context implicitly.

| Latest Model | Release Date | Details |
|--------------|--------------|---------|
| **Gemma 3** | Mar 12, 2025 | 1024-token window, 5:1 local/global ratio |

**Technical Notes**:
- Gemma 3's 5:1 ratio reduces attention compute by ~5x
- KV cache reduced to 0.78 GB (from 17 GB with MHA)

**Triton Implementations**:
| Repository | Description | Code Pointer |
|------------|-------------|--------------|
| **vLLM Prefill Attention** | SWA via `SLIDING_WINDOW_Q/K` params | [`vllm/v1/attention/ops/triton_prefill_attention.py:36`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_prefill_attention.py#L36) `_fwd_kernel` (lines 189-246 handle SWA) |
| **MaxLSB/flash-attn2** | FlashAttention-2 with SWA (MODE==2) | [`src/flash_attention/forward.py:21`](https://github.com/MaxLSB/flash-attn2/blob/main/src/flash_attention/forward.py#L21) `_attn_fwd`, `:260` `_attn_fwd_inner` |
| **MaxLSB/flash-attn2 BWD** | SWA backward pass | [`src/flash_attention/backward.py:76`](https://github.com/MaxLSB/flash-attn2/blob/main/src/flash_attention/backward.py#L76) `_attn_bwd_dk_dv`, `:220` `_attn_bwd_dq` |

---

### KDA (Kimi Delta Attention)

**Concept**: Linear attention variant that extends Gated DeltaNet with channel-wise forget gates. Enables O(n) scaling with sequence length.

| Latest Model | Release Date | Details |
|--------------|--------------|---------|
| **Kimi Linear** | Oct 2025 | 3:1 KDA:Global ratio, 48B/3B active |

**Technical Notes**:
- Channel-wise (not head-wise) forget gates for fine-grained control
- 75% KV cache reduction vs full attention
- 6.3x decoding throughput improvement
- Works especially well for 1M+ token contexts

**Triton Implementations**:
| Repository | Description | Code Pointer |
|------------|-------------|--------------|
| **flash-linear-attention (KDA)** | Official KDA kernels (12 total) | [`fla/ops/kda/fused_recurrent.py:25`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/fused_recurrent.py#L25) `fused_recurrent_kda_fwd_kernel` |
| **flash-linear-attention (KDA chunk)** | Chunk-wise KDA kernels | [`fla/ops/kda/chunk_intra.py:36`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/chunk_intra.py#L36) `chunk_kda_fwd_kernel_inter_solve_fused`, `:360` `chunk_kda_bwd_kernel_intra` |
| **flash-linear-attention (KDA gate)** | KDA gate computation | [`fla/ops/kda/gate.py:81`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/gate.py#L81) `kda_gate_fwd_kernel`, `:137` `kda_gate_bwd_kernel` |
| **SGLang KDA** | KDA integration with SGLang | [`sglang/srt/layers/attention/fla/kda.py:158`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/fla/kda.py#L158) (7 kernels) |

---

### Gated Attention

**Concept**: Query-dependent sparse gate after SDPA output, eliminating attention sink phenomenon.

| Latest Model | Release Date | Details |
|--------------|--------------|---------|
| **Qwen (NeurIPS 2025 Best Paper)** | 2025 | Attention sink reduced from 46.7% → 4.8% |

**Technical Notes**:
- Sigmoid gating per head
- Non-linearity in low-rank transformation
- Enables larger learning rates

**Triton Implementations**:
| Repository | Description | Code Pointer |
|------------|-------------|--------------|
| **gated_attention** | Official NeurIPS 2025 implementation | PyTorch-only (no Triton kernels); uses standard `torch.nn` with FlashAttention backend |

---

## 2. Linear Attention Variants

### Gated DeltaNet

**Concept**: Linear attention with gated memory updates using delta rule for efficient recurrent computation.

| Latest Model | Release Date | Details |
|--------------|--------------|---------|
| **Qwen3-Next** | Sep 10, 2025 | 3:1 linear:global ratio, 80B total/3B active |

**Technical Notes**:
- Chunkwise parallelism for efficient training
- Data-dependent gating for improved memory management
- Fast-weight RNN formulation
- Base mechanism that KDA extends with channel-wise gating

**Triton Implementations**:
| Repository | Description | Code Pointer |
|------------|-------------|--------------|
| **GatedDeltaNet (Official NVIDIA)** | ICLR 2025 official (19 kernels) | [`lit_gpt/gated_delta_rule_ops/chunk.py:102`](https://github.com/NVlabs/GatedDeltaNet/blob/main/lit_gpt/gated_delta_rule_ops/chunk.py#L102) `chunk_gated_delta_rule_fwd_kernel_h`, `:251` `chunk_gated_delta_rule_bwd_kernel_dhu` |
| **GatedDeltaNet (FLA version)** | FLA-compatible chunk kernels | [`lit_gpt/gated_delta_rule_ops/fla_version/chunk_fla.py:588`](https://github.com/NVlabs/GatedDeltaNet/blob/main/lit_gpt/gated_delta_rule_ops/fla_version/chunk_fla.py#L588) `chunk_gated_delta_rule_fwd_kernel_h` |
| **flash-linear-attention** | Community Triton with H100 opts | [`fla/ops/gated_delta_rule/fused_recurrent.py:20`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/fused_recurrent.py#L20), [`fla/ops/gated_delta_rule/wy_fast.py:22`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/wy_fast.py#L22) |
| **Helion** | Gated DeltaNet fwd_h kernel in Helion DSL | [`examples/gdn_fwd_h.py:28`](https://github.com/pytorch-labs/helion/blob/main/examples/gdn_fwd_h.py#L28) `helion_gdn_fwd_h` |

---

### RetNet / RWKV-6 / RWKV-7

**Concept**: Recurrent linear attention with exponential decay (RetNet) or receptance-weighted key-value (RWKV). Constant-size cache with linear scaling.

| Latest Model | Release Date | Details |
|--------------|--------------|---------|
| **RWKV-7** | 2025 | 1.37x speedup over FA3 at 128K tokens |

**Technical Notes**:
- Constant-size cache enables infinite context
- Parallel/chunked processing modes
- RWKV-X achieves competitive performance with Transformers

**Triton Implementations**:
| Repository | Description | Code Pointer |
|------------|-------------|--------------|
| **flash-linear-attention (RWKV6)** | RWKV-6 fused recurrent | [`fla/ops/rwkv6/fused_recurrent.py:26`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/rwkv6/fused_recurrent.py#L26), [`fla/ops/rwkv6/chunk.py:40`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/rwkv6/chunk.py#L40) (8 kernels) |
| **flash-linear-attention (RWKV7)** | RWKV-7 with channel mixing | [`fla/ops/rwkv7/fused_recurrent.py:30`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/rwkv7/fused_recurrent.py#L30), [`fla/ops/rwkv7/channel_mixing.py:33`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/rwkv7/channel_mixing.py#L33) |
| **flash-linear-attention (RWKV4)** | RWKV-4 recurrent | [`fla/ops/rwkv4/fused_recurrent.py:23`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/rwkv4/fused_recurrent.py#L23) |

---

### GLA (Gated Linear Attention)

**Concept**: Linear attention with data-dependent gating for improved memory and expressiveness. Core mechanism extended by Gated DeltaNet and KDA.

| Latest Model | Release Date | Details |
|--------------|--------------|---------|
| **GLA-based architectures** | 2024-2025 | Foundation for modern linear attention |

**Technical Notes**:
- Gated recurrent computation with linear complexity
- Chunk-wise parallelism for efficient training
- Foundation architecture for KDA and Gated DeltaNet extensions

**Triton Implementations**:
| Repository | Description | Code Pointer |
|------------|-------------|--------------|
| **flash-linear-attention (GLA chunk)** | Chunk-wise GLA kernels | [`fla/ops/gla/chunk.py:28`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gla/chunk.py#L28) `chunk_gla_fwd_kernel_o`, `:150` `chunk_gla_bwd_kernel_dv` |
| **flash-linear-attention (GLA fused)** | Fused recurrent GLA | [`fla/ops/gla/fused_recurrent.py:20`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gla/fused_recurrent.py#L20) `fused_recurrent_gla_fwd_kernel` |
| **flash-linear-attention (common chunk output)** | Common chunk output kernel for linear attention | [`fla/ops/common/chunk_o.py:30`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/common/chunk_o.py#L30) `chunk_fwd_kernel_o` |
| **flash-linear-attention (triangular solve)** | Lower triangular solve for linear attention | [`fla/ops/utils/solve.py:23`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/solve.py#L23) `solve_tril_16x16_kernel` |
| **vLLM FLA (scaled dot KKT)** | Scaled dot product for FLA | [`vllm/.../fla/ops/chunk_scaled_dot.py`](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/fla/ops) `chunk_scaled_dot_kkt_fwd_kernel` |

---

### Mamba / Mamba-2

**Concept**: State-space models as input-conditioned RNNs with selective gating.

| Latest Model | Release Date | Details |
|--------------|--------------|---------|
| **Mamba-2** | 2024 | 5x higher throughput than Transformers |

**Technical Notes**:
- Hardware-aware parallel scan
- Selective state propagation
- Linear scaling with sequence length

**Triton Implementations**:
| Repository | Description | Code Pointer |
|------------|-------------|--------------|
| **mamba-ssm (Official)** | Official Mamba-2 SSD kernels (32 total) | [`mamba_ssm/ops/triton/ssd_chunk_scan.py:48`](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/ssd_chunk_scan.py#L48) `_chunk_scan_fwd_kernel` |
| **mamba-ssm (state passing)** | State passing kernels | [`mamba_ssm/ops/triton/ssd_state_passing.py:29`](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/ssd_state_passing.py#L29) `_state_passing_fwd_kernel` |
| **mamba-ssm (chunk state)** | Chunk state kernels | [`mamba_ssm/ops/triton/ssd_chunk_state.py:191`](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/ssd_chunk_state.py#L191) `_chunk_state_fwd_kernel` |
| **mamba-ssm (selective scan)** | Selective scan update | [`mamba_ssm/ops/triton/selective_state_update.py:23`](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/selective_state_update.py#L23) `_selective_scan_update_kernel` |
| **mamba-ssm (causal conv1d)** | Causal convolution | [`mamba_ssm/ops/triton/causal_conv1d.py`](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/causal_conv1d.py) `_causal_conv1d_fwd_kernel` |
| **vLLM Mamba ops** | Adapted Mamba kernels | [`vllm/model_executor/layers/mamba/ops/`](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/mamba/ops) (7 kernel files) |
| **SGLang Mamba ops** | Adapted from vLLM | [`sglang/srt/layers/attention/mamba/`](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/layers/attention/mamba) |

---

## 3. Sparse Attention Variants

### NSA (Native Sparse Attention)

**Concept**: Hardware-aligned sparse attention with three-branch architecture: compressed (coarse), selected (fine), and sliding (local).

| Latest Model | Release Date | Details |
|--------------|--------------|---------|
| **DeepSeek V3.2** | 2025 | Production sparse attention, ACL 2025 Best Paper |

**Technical Notes**:
- Three branches: Compressed + Selected + Sliding
- Designed for 64K+ sequences
- Surpasses full attention on long-context benchmarks
- Foundation for DeepSeek Sparse Attention (DSA)

**Triton Implementations**:
| Repository | Description | Code Pointer |
|------------|-------------|--------------|
| **native-sparse-attention-triton** | 24 kernels, 29ms vs 810ms FlashAttn | [`ops/triton/topk_sparse_attention.py:26`](https://github.com/XunhaoLai/native-sparse-attention-triton/blob/main/native_sparse_attention/ops/triton/topk_sparse_attention.py#L26) `forward_kernel`, `:480` `backward_dkdv` |
| **native-sparse-attention-triton (compressed)** | Compressed attention branch | [`ops/triton/compressed_attention.py:27`](https://github.com/XunhaoLai/native-sparse-attention-triton/blob/main/native_sparse_attention/ops/triton/compressed_attention.py#L27) `forward_kernel`, `:851` `score_kernel` |
| **native-sparse-attention (fla-org)** | Full training support (12 kernels) | [`ops/parallel.py:471`](https://github.com/fla-org/native-sparse-attention/blob/main/native_sparse_attention/ops/parallel.py#L471) `parallel_nsa_fwd_kernel`, `:338` `parallel_nsa_kernel_topk` |
| **nsa-impl** | Selection attention kernels | [`nsa/selection.py:35`](https://github.com/tilde-research/nsa-impl/blob/main/nsa/selection.py#L35) `_sel_attn_fwd_kernel`, `:197` `_sel_attn_bwd_kernel` |
| **SGLang NSA** | NSA integration | [`sglang/srt/layers/attention/nsa/triton_kernel.py:9`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/nsa/triton_kernel.py#L9) |

---

### Block Sparse Attention

**Concept**: Mixed sparse patterns with streaming and block granularity.

| Latest Model | Release Date | Details |
|--------------|--------------|---------|
| **FlashAttention 2.4+** | 2024-2025 | Integrated block sparse support |

**Technical Notes**:
- Dynamic pruning via softmax thresholding (BLASST)
- Configurable block sizes for different workloads

**Triton Implementations**:
| Repository | Description | Code Pointer |
|------------|-------------|--------------|
| **MIT HAN Lab Block-Sparse-Attention** | CUDA kernels (no Triton) | [`csrc/block_sparse_attn/src/flash_fwd_block_*.cu`](https://github.com/mit-han-lab/Block-Sparse-Attention/tree/main/csrc/block_sparse_attn/src) (CUDA C++, not Triton) |
| **SGLang Double Sparsity** | Triton double sparsity attention | [`sglang/srt/layers/attention/triton_ops/double_sparsity_attention.py:22`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/triton_ops/double_sparsity_attention.py#L22) (7 kernels) |

---

## 4. Quantized Attention

### SageAttention (INT8/FP8/FP4)

**Concept**: Quantized attention with INT8 QK^T and FP8/FP4 PV computation.

| Latest Model | Release Date | Details |
|--------------|--------------|---------|
| **SageAttention3** | 2025 | 1038 TOPS with FP4 on RTX5090, INT8 training support |

**Technical Notes**:
- Per-block quantization with smoothing for Q/K matrices
- 2-5x speedup vs FlashAttention
- ICLR/ICML/NeurIPS 2025 papers

**Triton Implementations**:
| Repository | Description | Code Pointer |
|------------|-------------|--------------|
| **SageAttention (INT8 quant)** | Per-block INT8 quantization | [`sageattention/triton/quant_per_block.py:21`](https://github.com/thu-ml/SageAttention/blob/main/sageattention/triton/quant_per_block.py#L21) `quant_per_block_int8_kernel` |
| **SageAttention (INT8 attn)** | INT8 QK attention forward | [`sageattention/triton/attn_qk_int8_per_block.py:75`](https://github.com/thu-ml/SageAttention/blob/main/sageattention/triton/attn_qk_int8_per_block.py#L75) `_attn_fwd`, `:21` `_attn_fwd_inner` |
| **SageAttention (INT8 causal)** | Causal INT8 attention | [`sageattention/triton/attn_qk_int8_per_block_causal.py:68`](https://github.com/thu-ml/SageAttention/blob/main/sageattention/triton/attn_qk_int8_per_block_causal.py#L68) `_attn_fwd` |
| **SageAttention (INT4 quant)** | INT4 query/key quantization | [`sageattention/triton/quant_per_thread.py:100`](https://github.com/thu-ml/SageAttention/blob/main/sageattention/triton/quant_per_thread.py#L100) `quant_query_per_thread_int4_kernel` |

---

## 5. MoE (Mixture of Experts)

### Fused MoE + FP8

**Concept**: Fused routing + grouped GEMM with FP8 quantization for efficient expert computation.

| Latest Model | Release Date | Details |
|--------------|--------------|---------|
| **DeepSeek-V3** | 2025 | 256 experts, top-8 routing |
| **Llama-4** | Apr 2025 | MoE architecture |

**Technical Notes**:
- Online FP8 quantization during routing
- Pipelined group GEMM
- Align & sort fusion (token permutation + expert computation)
- Single-GPU kernel, uses Expert Parallelism for multi-GPU

**Triton Implementations**:
| Repository | Description | Code Pointer |
|------------|-------------|--------------|
| **vLLM fused_moe** | Production FP8 MoE kernel | [`vllm/model_executor/layers/fused_moe/fused_moe.py:314`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py#L314) `fused_moe_kernel`, `:81` `fused_moe_kernel_gptq_awq` |
| **SGLang fused_moe_triton** | Fused MoE with activations | [`sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py:334`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py#L334) `fused_moe_kernel`, `:865` `act_and_mul_kernel` |
| **SGLang MoE router** | Router kernels (CUDA/Tensor core) | [`sglang/srt/layers/moe/router.py:13`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/router.py#L13) `fused_moe_router_cudacore_kernel`, `:159` `fused_moe_router_tensorcore_kernel` |
| **SGLang EP MoE** | Expert Parallel MoE (24 kernels) | [`sglang/srt/layers/moe/ep_moe/kernels.py:73`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/ep_moe/kernels.py#L73) `deepep_permute_triton_kernel`, `:790` `_fwd_kernel_ep_gather` |
| **SGLang DeepEP Post-Reorder** | Output reordering for DeepEP | [`sglang/srt/layers/moe/ep_moe/kernels.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/ep_moe/kernels.py) `deepep_post_reorder_triton_kernel` |
| **vLLM EP Scatter/Gather** | Expert parallelism distribution | [`vllm/.../fused_moe.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py) `_fwd_kernel_ep_scatter_1/2`, `_fwd_kernel_ep_gather` |
| **vLLM Compute Identity** | Expert identity computation | [`vllm/.../fused_moe.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py) `compute_identity_kernel` |

---

## 6. Paged Attention (Prefill/Decode)

### Paged KV-Cache Attention

**Concept**: Memory-efficient attention with paged KV cache, supporting both prefill (initial sequence processing) and decode (autoregressive generation) phases.

| Latest Model | Release Date | Details |
|--------------|--------------|---------|
| **All decoder LLMs** | 2024-2025 | Universal infrastructure for LLM serving |

**Technical Notes**:
- Paged memory allocation eliminates fragmentation
- Unified kernel handles arbitrary query lengths (prefill + decode in same batch)
- POD-Attention: Fused prefill+decode for mixed batching scenarios
- vLLM V1 Triton kernel outperforms FlashAttention backend on H100

**Triton Implementations**:
| Repository | Description | Code Pointer |
|------------|-------------|--------------|
| **vLLM Triton Decode** | Paged decode attention (2-stage) | [`vllm/v1/attention/ops/triton_decode_attention.py:58`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_decode_attention.py#L58) `_fwd_kernel_stage1`, `:494` `_fwd_kernel_stage2` |
| **vLLM Triton Prefill** | Paged prefill attention | [`vllm/v1/attention/ops/triton_prefill_attention.py:36`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_prefill_attention.py#L36) `_fwd_kernel` |
| **vLLM Unified Attention 2D** | Combined prefill+decode | [`vllm/v1/attention/ops/triton_unified_attention.py:56`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_unified_attention.py#L56) `kernel_unified_attention_2d` |
| **vLLM Unified Attention 3D** | Segmented attention (memory-efficient) | [`vllm/v1/attention/ops/triton_unified_attention.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_unified_attention.py) `kernel_unified_attention_3d` |
| **SGLang Extend Attention** | Extend/prefill attention | [`sglang/srt/layers/attention/triton_ops/extend_attention.py:219`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/triton_ops/extend_attention.py#L219) `_fwd_kernel`, `:691` `_fwd_kernel_unified` |
| **SGLang Prefill Attention** | Simple prefill-only attention | [`sglang/srt/layers/attention/triton_ops/prefill_attention.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/triton_ops/prefill_attention.py) `_fwd_kernel` |
| **FlashInfer Paged** | Batch indices/positions | [`flashinfer/triton/page.py:21`](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/triton/page.py#L21) `get_batch_indices_positions_kernel` |

---

### Chunked Prefill Paged Decode

**Concept**: Hybrid kernel that combines chunked prefill and paged decode operations in a single fused kernel, enabling efficient mixed batching where some sequences are in prefill phase while others are decoding.

| Latest Model | Release Date | Details |
|--------------|--------------|---------|
| **All decoder LLMs with hybrid batching** | 2024-2025 | Enables mixing prefill and decode in same batch |

**Technical Notes**:
- Fuses prefill chunking with paged decode for memory efficiency
- Reduces kernel launch overhead for mixed workloads
- Supports sink tokens and attention biases
- Optimized for varying sequence lengths in the same batch

**Triton Implementations**:
| Repository | Description | Code Pointer |
|------------|-------------|--------------|
| **vLLM Chunked Prefill Paged Decode** | Hybrid prefill+decode kernel | [`vllm/v1/attention/ops/chunked_prefill_paged_decode.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/chunked_prefill_paged_decode.py) `chunked_prefill_paged_decode_kernel` |

---

### Cascade Attention

**Concept**: Hierarchical KV-Cache for shared prefixes. Stores shared prefix KV in GPU shared memory for fast access across multiple requests.

| Latest Model | Release Date | Details |
|--------------|--------------|---------|
| **Shared-prefix batching** | 2024-2025 | Up to 31x speedup vs vLLM PageAttention |

**Technical Notes**:
- Decouples attention of shared prefix and unique suffixes
- Shared KV-Cache in SMEM enables fast multi-request access
- Block sparse format with (3,1) block size for 3-query KV sharing
- 26x speedup over FlashInfer batch decode without cascading

**Triton Implementations**:
| Repository | Description | Code Pointer |
|------------|-------------|--------------|
| **FlashInfer Cascade (state merge)** | Merges attention states | [`flashinfer/triton/kernels/cascade.py:24`](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/triton/kernels/cascade.py#L24) `merge_state_kernel`, `:61` `merge_state_in_place_kernel` |
| **FlashInfer Cascade (multi-state)** | Multiple state merging | [`flashinfer/triton/kernels/cascade.py:99`](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/triton/kernels/cascade.py#L99) `merge_states_kernel`, `:134` `variable_length_merge_states_kernel` |
| **vLLM Merge Attention States** | Attention state merging | [`vllm/v1/attention/ops/triton_merge_attn_states.py:44`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_merge_attn_states.py#L44) `merge_attn_states_kernel` |

**Reference**: [Cascade Inference Blog](https://flashinfer.ai/2024/02/02/cascade-inference.html)

---

## 7. Sampling Kernels

### Top-K / Top-P / Min-P Sampling

**Concept**: Sorting-free token selection algorithms that filter and sample from model output probabilities without expensive full-vocabulary sorting.

| Latest Model | Release Date | Details |
|--------------|--------------|---------|
| **All autoregressive LLMs** | 2024-2025 | Universal sampling infrastructure |

**Technical Notes**:
- **Inverse Transform Sampling**: CDF-based with BlockReduce + BlockScan + early stopping
- **Rejection Sampling**: Iterative pivot refinement
- **Dual-Pivot Rejection Sampling**: O(log(1/ε)) guaranteed convergence (FlashInfer v0.2.3+)
- **Top-K**: Sorting-free parallel selection, up to 10x speedup
- **Top-P (Nucleus)**: Cumulative probability threshold
- **Min-P**: Filter below `p_base × p_max`
- 50%+ reduction in sampling time vs sorting-based PyTorch

**Triton Implementations**:
| Repository | Description | Code Pointer |
|------------|-------------|--------------|
| **vLLM Triton Sampler** | Pivot-search top-k/top-p (10x speedup) | [vLLM PR #25824](https://github.com/vllm-project/vllm/pull/25824) - env: `VLLM_USE_TRITON_SAMPLER=1` |
| **vLLM Gumbel Sampling** | Gumbel-max trick sampling | [`vllm/v1/worker/gpu/sample/triton_sampler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/triton_sampler.py) `_gumbel_sample_kernel` |
| **vLLM Temperature** | Temperature scaling | [`vllm/v1/worker/gpu/sample/triton_sampler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/triton_sampler.py) `_temperature_kernel` |
| **vLLM Min-P** | Min-p filtering | [`vllm/v1/worker/gpu/sample/triton_sampler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/triton_sampler.py) `_min_p_kernel` |
| **vLLM Penalties** | Repetition/frequency penalties | [`vllm/v1/worker/gpu/sample/triton_sampler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/triton_sampler.py) `_penalties_kernel`, `_bincount_kernel` |
| **FlashInfer Sampling** | CUDA kernels (not Triton) | [`csrc/sampling.cu`](https://github.com/flashinfer-ai/flashinfer/tree/main/csrc) - dual-pivot rejection sampling |
| **vLLM Top-K Log Softmax** | Top-K selection with fused log softmax | [`vllm/v1/worker/gpu/sample/triton_sampler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/triton_sampler.py) `_topk_log_softmax_kernel` |

**Reference**: [FlashInfer Sampling Blog](https://flashinfer.ai/2025/03/10/sampling.html)

---

### Top-K Log Softmax

**Concept**: Fused kernel that combines top-k selection with log softmax computation, avoiding redundant passes over vocabulary.

| Latest Model | Release Date | Details |
|--------------|--------------|---------|
| **All autoregressive LLMs** | 2024-2025 | Optimized sampling with fused ops |

**Technical Notes**:
- Fuses top-k selection and log softmax into single kernel
- Reduces memory bandwidth by avoiding intermediate storage
- Uses parallel reduction for efficient top-k extraction
- Part of vLLM's Triton-based sampler (enabled via `VLLM_USE_TRITON_SAMPLER=1`)

**Triton Implementations**:
| Repository | Description | Code Pointer |
|------------|-------------|--------------|
| **vLLM Top-K Log Softmax** | Fused top-k + log softmax | [`vllm/v1/worker/gpu/sample/triton_sampler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/triton_sampler.py) `_topk_log_softmax_kernel` |

---

### Speculative Decoding

**Concept**: Use a small draft model to propose multiple tokens, then verify with the target model in parallel. Reduces decoding latency by accepting multiple tokens per forward pass.

| Latest Model | Release Date | Details |
|--------------|--------------|---------|
| **EAGLE-2** | 2024-2025 | Dynamic draft trees, confidence-based pruning |
| **Medusa** | 2024 | Multiple prediction heads |
| **SpecInfer** | 2024 | Tree-based speculation |

**Technical Notes**:
- Draft model proposes K tokens, target model verifies in single forward pass
- Tree-based speculation (SpecInfer, EAGLE-2) organizes predictions hierarchically
- Chain speculative sampling for verification
- EAGLE-2: Dynamic trees based on context, pruning via confidence scores

**Triton Implementations**:
| Repository | Description | Code Pointer |
|------------|-------------|--------------|
| **vLLM EAGLE** | EAGLE prepare_inputs kernels | [`vllm/v1/spec_decode/utils.py:6`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/utils.py#L6) `eagle_prepare_inputs_padded_kernel`, `:49` `eagle_prepare_next_token_padded_kernel` |
| **SGLang EAGLE Info V2** | EAGLE cache management (4 kernels) | [`sglang/srt/speculative/eagle_info_v2.py:53`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/speculative/eagle_info_v2.py#L53) `assign_draft_cache_locs_page_size_1` |
| **SGLang Spec Utils** | Speculative decoding utilities (7 kernels) | [`sglang/srt/speculative/spec_utils.py:61`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/speculative/spec_utils.py#L61) `create_extend_after_decode_spec_info`, `:141` `assign_draft_cache_locs` |
| **SGLang Multi-Layer EAGLE** | Multi-layer EAGLE utilities | [`sglang/srt/speculative/multi_layer_eagle_utils.py:20`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/speculative/multi_layer_eagle_utils.py#L20) `rotate_input_ids_kernel` |
| **vLLM Rejection Sampling** | Simple linear rejection sampling | [`vllm/v1/worker/gpu/spec_decode/rejection_sample.py:9`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/spec_decode/rejection_sample.py#L9) `_rejection_sample_kernel` |

---

## 8. Quantization Kernels

### AWQ (Activation-aware Weight Quantization)

**Concept**: 4-bit weight quantization with per-group scales, preserving salient weights for minimal accuracy loss.

| Latest Model | Release Date | Details |
|--------------|--------------|---------|
| **All AWQ-quantized models** | 2024-2025 | Standard 4-bit inference |

**Technical Notes**:
- Per-group dequantization with INT4 weights and FP16 scales
- Fused dequant + GEMM for memory bandwidth optimization
- SGLang implementation adapted from vLLM

**Triton Implementations**:
| Repository | Description | Code Pointer |
|------------|-------------|--------------|
| **vLLM AWQ Dequantize** | 4-bit weight unpacking | [`vllm/model_executor/layers/quantization/kernels/triton/awq_triton.py:31`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/kernels/triton/awq_triton.py#L31) `awq_dequantize_kernel` |
| **vLLM AWQ GEMM** | Fused dequant + matmul | [`vllm/model_executor/layers/quantization/kernels/triton/awq_triton.py:76`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/kernels/triton/awq_triton.py#L76) `awq_gemm_kernel` |
| **SGLang AWQ** | Adapted from vLLM | [`sglang/srt/layers/quantization/awq_triton.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/awq_triton.py) |

---

### INT8 Block-wise Quantization

**Concept**: Per-token or per-block INT8 quantization for weights and activations with block-scaled matmul.

| Latest Model | Release Date | Details |
|--------------|--------------|---------|
| **DeepSeek-V3, Llama-4** | 2025 | W8A8 inference with block scaling |

**Technical Notes**:
- Per-token quantization: Each token has independent scale
- Per-group quantization: Groups of elements share scales
- Block-scaled matmul: Efficient INT8 GEMM with per-block scales

**Triton Implementations**:
| Repository | Description | Code Pointer |
|------------|-------------|--------------|
| **vLLM INT8 Per-Token Quant** | Token-wise quantization | [`vllm/model_executor/layers/quantization/kernels/triton/int8_kernel.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/kernels/triton/int8_kernel.py) `_per_token_quant_int8` |
| **vLLM INT8 Per-Group Quant** | Per-token-group INT8 quantization | [`vllm/model_executor/layers/quantization/utils/int8_utils.py:153`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/int8_utils.py#L153) `_per_token_group_quant_int8` |
| **vLLM INT8 Block Matmul** | Block-scaled INT8 GEMM | [`vllm/model_executor/layers/quantization/kernels/triton/int8_kernel.py:20`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/kernels/triton/int8_kernel.py#L20) `_w8a8_block_int8_matmul` |
| **SGLang INT8 Quant** | Per-token/group quantization | [`sglang/srt/layers/quantization/int8_kernel.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/int8_kernel.py) |

---

### FP8 Block-wise Quantization

**Concept**: FP8 (E4M3/E5M2) quantization with block-scaled matmul for reduced memory and compute.

| Latest Model | Release Date | Details |
|--------------|--------------|---------|
| **DeepSeek-V3, Llama-4** | 2025 | FP8 inference on H100/H200 |

**Technical Notes**:
- Per-token group quantization: Groups of tokens share FP8 scales
- Row-major vs column-major scale layouts for different GEMM patterns
- MxFP8: Microscaling format with `tl.dot_scaled()` intrinsic
- AMD-optimized: 4x unrolled kernels for HIP

**Triton Implementations**:
| Repository | Description | Code Pointer |
|------------|-------------|--------------|
| **vLLM FP8 Per-Token Quant** | Token-wise FP8 quantization | [`vllm/model_executor/layers/quantization/kernels/triton/fp8_kernel.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/kernels/triton/fp8_kernel.py) `_per_token_group_quant_fp8` |
| **vLLM FP8 Block Matmul** | Block-scaled FP8 GEMM | [`vllm/model_executor/layers/quantization/kernels/triton/fp8_kernel.py:28`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/kernels/triton/fp8_kernel.py#L28) `_w8a8_block_fp8_matmul` |
| **vLLM Scaled MM** | Generic scaled matmul | [`vllm/model_executor/layers/quantization/kernels/triton/triton_scaled_mm.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/kernels/triton/triton_scaled_mm.py) `scaled_mm_kernel` |
| **SGLang FP8 Quant** | Row/column-major scales | [`sglang/srt/layers/quantization/fp8_kernel.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/fp8_kernel.py) |
| **SGLang FP8 AMD Optimized** | 4x unrolled for AMD | [`sglang/srt/layers/quantization/fp8_kernel.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/fp8_kernel.py) `_w8a8_block_fp8_matmul_unrolledx4` |
| **SGLang MxFP8** | Microscaling with dot_scaled | [`sglang/srt/layers/quantization/fp8_kernel.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/fp8_kernel.py) `_mxfp8_block_scaled_matmul_kernel` |

---

## 9. LoRA Kernels

### SGMV (Segmented Grouped Matrix-Vector) LoRA

**Concept**: Efficient multi-adapter LoRA inference with batched low-rank matrix operations.

| Latest Model | Release Date | Details |
|--------------|--------------|---------|
| **All LoRA fine-tuned models** | 2024-2025 | Multi-tenant serving |

**Technical Notes**:
- **vLLM approach**: Unified multi-slice with 3D grid (M, slices, LoRA IDs)
- **SGLang approach**: Segment-based SGMV with layer specialization
- Split-K optimization for shrink operations
- Layer-specific kernels (QKV, MLP gate_up) for SGLang

**Triton Implementations**:
| Repository | Description | Code Pointer |
|------------|-------------|--------------|
| **vLLM LoRA Expand** | Latent → hidden mapping | [`vllm/lora/ops/triton_ops/lora_expand.py:21`](https://github.com/vllm-project/vllm/blob/main/vllm/lora/ops/triton_ops/lora_expand.py#L21) `_lora_expand_kernel` |
| **vLLM LoRA Shrink** | Hidden → latent mapping | [`vllm/lora/ops/triton_ops/lora_shrink.py:21`](https://github.com/vllm-project/vllm/blob/main/vllm/lora/ops/triton_ops/lora_shrink.py#L21) `_lora_shrink_kernel` |
| **vLLM LoRA Core Helpers** | Core expand/shrink operations | [`vllm/lora/ops/triton_ops/lora_expand.py`](https://github.com/vllm-project/vllm/blob/main/vllm/lora/ops/triton_ops/lora_expand.py) `do_expand_kernel`, [`lora_shrink.py`](https://github.com/vllm-project/vllm/blob/main/vllm/lora/ops/triton_ops/lora_shrink.py) `do_shrink_kernel` |
| **SGLang Chunked LoRA Expand** | Segmented expand | [`sglang/srt/lora/triton_ops/lora_expand.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/triton_ops/lora_expand.py) `_chunked_lora_expand_kernel` |
| **SGLang Chunked LoRA Shrink** | Segmented shrink | [`sglang/srt/lora/triton_ops/lora_shrink.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/triton_ops/lora_shrink.py) `_chunked_lora_shrink_kernel` |
| **SGLang QKV LoRA B** | QKV-specific kernel | [`sglang/srt/lora/triton_ops/lora_ops.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/triton_ops/lora_ops.py) `_qkv_lora_b_kernel` |
| **SGLang Gate-Up LoRA B** | MLP-specific kernel | [`sglang/srt/lora/triton_ops/lora_ops.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/triton_ops/lora_ops.py) `_gate_up_lora_b_kernel` |
| **SGLang SGEMM LoRA A/B** | Layer-specific matmul | [`sglang/srt/lora/triton_ops/lora_ops.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/triton_ops/lora_ops.py) `_sgemm_lora_a_kernel`, `_sgemm_lora_b_kernel` |

---

## 10. Normalization and Activation Kernels

### RMSNorm / LayerNorm

**Concept**: Layer normalization variants used throughout transformer architectures.

| Latest Model | Release Date | Details |
|--------------|--------------|---------|
| **All transformer LLMs** | 2024-2025 | Standard normalization layers |

**Technical Notes**:
- RMSNorm: Root mean square normalization (Llama family)
- LayerNorm: Full mean + variance normalization
- Fused variants with residual addition and quantization

**Triton Implementations**:
| Repository | Description | Code Pointer |
|------------|-------------|--------------|
| **vLLM RMSNorm** | Triton RMS normalization | [`vllm/model_executor/layers/quantization/kernels/triton/fp8_kernel.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/kernels/triton/fp8_kernel.py) (fused with quant) |
| **Mamba Layer Norm** | Gated layer norm for SSM | [`mamba_ssm/ops/triton/layer_norm.py`](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/layer_norm.py) `_layer_norm_fwd_1pass_kernel` |
| **vLLM Mamba LayerNorm** | Adapted Mamba layer norm | [`vllm/model_executor/layers/mamba/ops/layer_norm.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/ops/layer_norm.py) |

---

### SiLU + Mul (SwiGLU) Activation

**Concept**: Fused SiLU(x) * y activation used in MLP layers of modern LLMs.

| Latest Model | Release Date | Details |
|--------------|--------------|---------|
| **Llama, Mistral, DeepSeek** | 2024-2025 | Standard MLP activation |

**Technical Notes**:
- SwiGLU: SiLU(Wx) * (Vx) gating mechanism
- Often fused with quantization for memory efficiency
- MoE kernels include fused activation

**Triton Implementations**:
| Repository | Description | Code Pointer |
|------------|-------------|--------------|
| **vLLM Fused Activation + Quant** | SiLU + mul + FP8 quant | [`vllm/model_executor/layers/quantization/kernels/triton/fp8_kernel.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/kernels/triton/fp8_kernel.py) `_silu_mul_per_token_group_quant_fp8_colmajor` |
| **SGLang Act and Mul** | Activation + multiply | [`sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py:865`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py#L865) `act_and_mul_kernel` |

---

## Sources

- [DeepSeek R1 Technical Report](https://arxiv.org/abs/2501.12948)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [Gemma 3 Explained](https://developers.googleblog.com/gemma-explained-whats-new-in-gemma-3/)
- [Llama 4 Blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
- [NSA Paper](https://arxiv.org/abs/2502.11089)
- [Kimi Linear Paper](https://arxiv.org/pdf/2510.26692)
- [FlashInfer Documentation](https://docs.flashinfer.ai/)
- [FlashInfer MLSys 2025 Paper](https://arxiv.org/pdf/2501.01005)
- [FlashInfer Cascade Inference Blog](https://flashinfer.ai/2024/02/02/cascade-inference.html)
- [FlashInfer Sampling Blog](https://flashinfer.ai/2025/03/10/sampling.html)
- [MLSys 2026 FlashInfer Contest](https://mlsys26.flashinfer.ai/)
