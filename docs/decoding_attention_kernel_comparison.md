# Helion vs. Triton & cute-DSL Decoding Attention

## Decomposition & Launch Shape
- **Helion** (`examples/decoding_attention.py`:10) flattens `(batch, head)` into a single tiled loop, streaming KV cache blocks explicitly in software-managed tiles. Parallelism is described via Helion’s loop tiling rather than explicit threadblock launches.
- **Triton** (`benchmarks/tritonbench/tritonbench/operators/decoding_attention/operator.py`:150+) emits hand-tuned GPU kernels that launch grids across batches, heads, and split-K partitions. Online softmax and GEMM are fused, with split-K reductions handled on device.
- **cute-DSL** (`flash_attn.cute.interface.flash_attn_func`) expresses the same algorithm using CUTLASS-style tensor tiles; it still becomes highly parallel GPU kernels, but schedules are derived from cute’s tensor DSL.

## Attention Math & Numerical Stability
- All three implementations perform streaming softmax with running `m_i`/`l_i` accumulators.
- Helion mirrors Triton’s base-2 exponentiation (`_LOG2E` scaling) and masks invalid KV columns in the same way as Triton’s masked `exp2` path.
- The cute-DSL backend inherits identical stability logic from FlashAttention v2/v3.

## Head Mapping / GQA
- Helion maps query heads to KV heads after flattening using integer arithmetic (`examples/decoding_attention.py`:53-62), currently running on the host.
- Triton/xFormers compute the mapping inside the kernel with packed tensor layouts and optional `pack_gqa` handling.
- cute-DSL handles head packing internally; callers pass original tensors and the backend performs expansion.

## KV Cache Handling
- Helion’s example operates on dense per-sequence caches with scalar valid-length masks; paged caches and block tables are not yet present.
- Triton kernels support paged/block-table caches and additive biases (see `_pack_xformer_input` and paged mask logic around lines 250–330).
- cute-DSL backend also supports packed GQA and causal masks within its FlashAttention integration.

## Quantization Paths
- Helion currently assumes FP16/BF16 storage with FP32 accumulation; quantized KV caches are not implemented.
- Triton provides FP8 and int4 KV dequantization paths (`triton_splitk_fp8kv`, `fa3_kvcache_fp8qkv`).
- cute-DSL can consume FP8 caches through FlashAttention v3.

## Autotuning & Configurations
- Triton and cute-DSL register multiple kernel variants with autotuned block sizes and split-K strategies.
- Helion’s kernel presently uses a single schedule and relies on Helion’s autotuning framework once integrated.

## Summary
The Helion kernel matches Triton/cute-DSL at the algorithmic level—streaming softmax, causal masking, GQA mapping—but the existing Triton and cute-DSL backends expose additional capabilities (paged caches, quantized KV, split-K parallelism) and leverage GPU-specialized scheduling. Once the Helion framework issue with the flattened `(batch, head)` pattern is resolved, we can extend the Helion kernel to cover those extra features and close the functionality gap.
