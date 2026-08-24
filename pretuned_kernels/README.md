# Helion Pretuned Kernels

This directory contains pretuned Helion kernels, benchmark shape sweeps, and
checked-in AOT heuristic files.  They are meant to be useful copy/paste starting
points for common kernel patterns while also being runnable examples for people
who want to quickly try Helion.

The checked-in heuristics let these kernels run immediately without online
autotuning.  Heuristics ship for both NVIDIA H100 (`sm90`) and B200 (`sm100`);
Helion picks the matching file at runtime.  Treat the files as kernel recipes:
copy the kernel and its local `_helion_aot_*` heuristic into your code, then
retune when your target shapes or hardware differ materially from the included
sweep.

Each kernel module has a `main()` that benchmarks against one or more named
reference baselines.

## File structure

```
pretuned_kernels/
├── README.md
├── vector_add/
│   ├── vector_add.py                          # the kernel + main()
│   ├── _helion_aot_vector_add_cuda_sm100.py   # B200 heuristic
│   └── _helion_aot_vector_add_cuda_sm90.py    # H100 heuristic
├── softmax/
├── layer_norm/
├── rms_norm/
├── cross_entropy/
├── rope/
├── scaled_mm/
├── scale_mm_cute/                    # B200 CuTe (tcgen05) rowwise FP8 GEMM
├── grouped_gemm/                     # B200 grouped FP16 GEMM vs CuTeDSL
├── grouped_gemm_deepgemm/            # B200 grouped BF16 vs DeepGEMM
├── nvfp4_gemv/                       # B200 Triton NVFP4 (W4A4 / W4A16) decode GEMV
├── nvfp4_gemv_cute/                  # B200 Helion CuTe NVFP4 decode GEMV
├── projection_rotary/                # B200 CuTe projection + rotary fragment epilogue
├── interleaved_swiglu/               # B200 CuTe projection + compact SwiGLU epilogue
├── silu_mul_fp8/                     # ported from vLLM (vllm/kernels/helion/ops)
├── dynamic_per_token_scaled_fp8_quant/
├── per_token_group_fp8_quant/
├── rms_norm_dynamic_per_token_quant/
├── rms_norm_per_block_quant/
├── silu_and_mul_per_block_quant/
├── fused_qk_norm_rope/
├── causal_conv1d/                    # TPU/Pallas fixed-config decode kernel
└── gdn_decode/                       # TPU/Pallas fixed-config recurrent decode
```

Each kernel ships with one heuristic file per supported compute capability.
At runtime Helion picks the file matching the current GPU.

| Kernel | Shape sweep | Reference baseline |
|---|---|---|
| `vector_add` | `2**i for i in range(19, 29)` | `x + y` |
| `softmax` | Triton tutorial `M=4096, N=128*i for i in range(2, 100)` + realistic long-context shapes | `F.softmax` |
| `layer_norm` | Triton tutorial `M=4096, N=512*i for i in range(2, 32)` + realistic hidden-size shapes | `F.layer_norm` |
| `rms_norm` | TritonBench `(M=2048, H)` default + NPOT shapes + realistic LLM hidden-size and production-style shapes | `F.rms_norm` |
| `cross_entropy` | TritonBench/Liger token-vocab sweep + realistic LLM vocabulary shapes | `F.cross_entropy` |
| `rope` | TritonBench RoPE `(H, T)` defaults with exact shape buckets and `H8192_T2048` fallback | eager RoPE reference |
| `scaled_mm` | vLLM Qwen3 FP8 `(K, N)` weight shapes at small token counts `M in {16, 64}` | `torch._scaled_mm` |
| `scale_mm_cute` | Skinny-M FP8 decode + decoder-layer FP8 W8A8 serving `(M, K, N)` shapes (B200 CuTe backend only) | `torch._scaled_mm` (rowwise) + vLLM CUTLASS |
| `grouped_gemm` | Seven CUTLASS-example-derived heterogeneous FP16 grouped-NT cases (3--4 GEMMs, including M/N tails; B200 CuTe only) | pinned NVIDIA CUTLASS CuTeDSL kernel |
| `grouped_gemm_deepgemm` | Eight official DeepGEMM BF16 grouped-NT shapes with deterministic heterogeneous per-group M (B200 CuTe only) | pinned DeepGEMM `m_grouped_bf16_gemm_nt_contiguous` public API (K-major B, `compiled_dims="nk"`, non-PSUM) |
| `nvfp4_gemv` | Decode (M=1) NVFP4 GEMV `(N, K)` weight shapes (Llama-3 / Qwen projections), W4A4 + W4A16 (B200 Triton backend) | NVFP4 dequant reference + vLLM CUTLASS `cutlass_scaled_fp4_mm` |
| `nvfp4_gemv_cute` | Decode (M=1) NVFP4 GEMV `(N, K)` weight shapes, W4A4 + W4A16, using Helion kernels compiled with the CuTe backend on B200 | NVFP4 dequant reference + vLLM CUTLASS `cutlass_scaled_fp4_mm` |
| `projection_rotary` | BF16 `(M=1024, K=4096, heads=32, D=128)` projection with fused bias and adjacent-pair rotary mixing (B200 CuTe only) | eager BF16 projection + rotary composition |
| `interleaved_swiglu` | BF16 `(M=1024, K=4096, heads=1, packed D=11008)` projection with interleaved gate/value columns (B200 CuTe only) | eager BF16 projection + SwiGLU composition |
| `silu_mul_fp8` | vLLM `(num_tokens, intermediate)` decode shapes | torch-native silu-and-mul + fp8 quant |
| `dynamic_per_token_scaled_fp8_quant` | vLLM `(num_tokens, hidden)` shapes | torch-native per-token fp8 quant |
| `per_token_group_fp8_quant` | vLLM `(num_tokens, hidden, group)` shapes | torch-native per-group fp8 quant |
| `rms_norm_dynamic_per_token_quant` | vLLM `(num_tokens, hidden)` shapes | torch-native RMSNorm + per-token fp8 quant |
| `rms_norm_per_block_quant` | vLLM `(num_tokens, hidden, group)` shapes | torch-native RMSNorm + per-block fp8 quant |
| `silu_and_mul_per_block_quant` | vLLM `(num_tokens, intermediate, group)` shapes | torch-native silu-and-mul + per-block fp8 quant |
| `fused_qk_norm_rope` | vLLM `(num_tokens, q_heads, kv_heads)` shapes | torch-native fused QK-RMSNorm + RoPE |
| `causal_conv1d` | TPU decode `N=512, H=4, D=128, W=4` | tpu-inference `ragged_causal_conv1d` |
| `gdn_decode` | TPU decode `N=512, H=2, K=V=128` | tpu-inference `fused_decoding_gdn` |

Most kernels additionally benchmark against `torch.compile` of the listed
PyTorch baseline (a speedup-comparison baseline only -- correctness is checked
against the eager reference). The grouped-GEMM entries instead use the named
CUDA references in the table. The headline speedup is Helion vs the *fastest*
available baseline, and the per-kernel dropdown reports every baseline.

`grouped_gemm` compares Helion with the same pinned CUTLASS kernel. Required
device pointer tables are initialized before graph capture, and both
implementations are checked against the same pointer-target outputs before
timing.

`grouped_gemm_deepgemm` is separate because it follows DeepGEMM's BF16 packed-A
layout and official shape suite rather than the CUTLASS FP16 heterogeneous-M/N/K
suite. Its checked-in B200 artifact selects reviewed `worklist_nm` configs by
the exact official seed-0 workload: logical shape, A/B/worklist dtypes and
storage layouts, source-M tile, packed work volume, and normalized worklist must
all match. The benchmark compares those profiles with one specific DeepGEMM
public path: contiguous K-major B, `compiled_dims="nk"`, non-PSUM,
upstream-recommended B200 alignment 224, and padding-zero work disabled. Packing
and compilation happen before timing. Both implementations replay pre-captured
graphs, clear L2 before every measurement, warm the GPU before each case, and
reverse measurement order to reduce clock and ordering bias.

The kernels ported from vLLM (`vllm/kernels/helion/ops`) benchmark each fused
Helion kernel under CUDA graphs against a torch-native (unfused, eager)
reference; `silu_mul_fp8` ships an `sm90` heuristic only, the rest ship both
`sm90` and `sm100`.

`causal_conv1d` and `gdn_decode` are Pallas kernels with checked-in fixed
configurations rather than CUDA AOT heuristic files. Their module-level
`main()` functions run real-TPU JAX-export correctness checks. They are not
registered in the CUDA-only aggregate runner. The implementations and
baselines follow the Apache-licensed `vllm-project/tpu-inference` state-cache
contracts; `gdn_decode` currently targets `H=2` because larger recurrent-state
tiles exceed Helion's current aligned indirect-DMA VMEM plan.

## Scope

Use this directory as a collection of pretuned kernels and runnable examples.
For production code, copy the relevant kernel pattern into the application.  If
the shapes or target hardware differ from the included sweep, generate and
commit an AOT heuristic for the application's target shapes and hardware.

For AOT pretuned kernels, Helion's runtime looks for
`_helion_aot_<kernel>_cuda_sm<NN>.py` next to the kernel source file.  Helion
looks for AOT heuristic files for the current compute capability first, then
falls back to older compatible CUDA/ROCm capabilities.  For example, on
`sm120`, if no `sm120` heuristic exists, an `sm100` heuristic file can be used.

## Running benchmarks

Each kernel module has a `main()` that benchmarks the Helion kernel against its
named reference baselines across the included shape set:

```bash
cd pretuned_kernels/softmax
python softmax.py
```

The external grouped-GEMM references need explicit pinned checkouts:

```bash
export HELION_CUTLASS_GROUPED_GEMM_SOURCE=/path/to/CUTLASS/.../grouped_gemm.py
python -m pretuned_kernels.grouped_gemm.grouped_gemm

export HELION_DEEPGEMM_ROOT=/path/to/built/DeepGEMM
python -m pretuned_kernels.grouped_gemm_deepgemm.grouped_gemm_deepgemm
```

The modules report the required commits when either variable is missing. The
DeepGEMM comparison verifies the source revisions, module origin, and native
ABI and records the extension hash; benchmark evidence should use a freshly
built checkout.

## Adding a heuristic for new hardware

These kernels ship pretuned heuristics for specific GPU architectures.  To add
another GPU, run the AOT autotune workflow on that hardware against the kernel;
the runner emits a new
`_helion_aot_<kernel>_<device>_<cc>.py` next to the kernel source, which
you commit alongside the existing one(s).  Helion picks the right one at
runtime based on the running GPU's compute capability (with fallback to
older compatible capabilities, e.g. `sm120` → `sm100`).

For a kernel whose `main()` benchmarks under CUDA graphs (`use_cudagraph()`
returns `True`, e.g. `scaled_mm` and the vLLM-ported ops), set
`HELION_BENCHMARK_CUDAGRAPH=1` for the autotune run so the autotuner benchmarks
candidate configs the same way — matching the deployment/benchmark timing
regime.

See the [Ahead-of-Time (AOT) Heuristic Tuning](../docs/deployment_autotuning.md#ahead-of-time-aot-heuristic-tuning)
section of `docs/deployment_autotuning.md` for the end-to-end workflow,
runner CLI, generated artifacts, and runtime fallback rules — including
a worked "Pretuning a kernel for new hardware" walkthrough.
