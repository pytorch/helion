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

Each kernel module has a `main()` that benchmarks against PyTorch eager.

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
├── silu_mul_fp8/                     # ported from vLLM (vllm/kernels/helion/ops)
├── dynamic_per_token_scaled_fp8_quant/
├── per_token_group_fp8_quant/
├── rms_norm_dynamic_per_token_quant/
├── rms_norm_per_block_quant/
├── silu_and_mul_per_block_quant/
└── fused_qk_norm_rope/
```

Each kernel ships with one heuristic file per supported compute capability.
At runtime Helion picks the file matching the current GPU.

| Kernel | Shape sweep | PyTorch baseline |
|---|---|---|
| `vector_add` | `2**i for i in range(19, 29)` | `x + y` |
| `softmax` | Triton tutorial `M=4096, N=128*i for i in range(2, 100)` + realistic long-context shapes | `F.softmax` |
| `layer_norm` | Triton tutorial `M=4096, N=512*i for i in range(2, 32)` + realistic hidden-size shapes | `F.layer_norm` |
| `rms_norm` | TritonBench `(M=2048, H)` default + NPOT shapes + realistic LLM hidden-size and production-style shapes | `F.rms_norm` |
| `cross_entropy` | TritonBench/Liger token-vocab sweep + realistic LLM vocabulary shapes | `F.cross_entropy` |
| `rope` | TritonBench RoPE `(H, T)` defaults with exact shape buckets and `H8192_T2048` fallback | eager RoPE reference |
| `scaled_mm` | vLLM Qwen3 FP8 `(K, N)` weight shapes at small token counts `M in {16, 64}` | `torch._scaled_mm` |
| `silu_mul_fp8` | vLLM `(num_tokens, intermediate)` decode shapes | torch-native silu-and-mul + fp8 quant |
| `dynamic_per_token_scaled_fp8_quant` | vLLM `(num_tokens, hidden)` shapes | torch-native per-token fp8 quant |
| `per_token_group_fp8_quant` | vLLM `(num_tokens, hidden, group)` shapes | torch-native per-group fp8 quant |
| `rms_norm_dynamic_per_token_quant` | vLLM `(num_tokens, hidden)` shapes | torch-native RMSNorm + per-token fp8 quant |
| `rms_norm_per_block_quant` | vLLM `(num_tokens, hidden, group)` shapes | torch-native RMSNorm + per-block fp8 quant |
| `silu_and_mul_per_block_quant` | vLLM `(num_tokens, intermediate, group)` shapes | torch-native silu-and-mul + per-block fp8 quant |
| `fused_qk_norm_rope` | vLLM `(num_tokens, q_heads, kv_heads)` shapes | torch-native fused QK-RMSNorm + RoPE |

Every kernel additionally benchmarks against `torch.compile` of the listed
PyTorch baseline (a speedup-comparison baseline only -- correctness is checked
against the eager reference). The headline speedup is Helion vs the *fastest*
available baseline, and the dashboard's per-kernel dropdown breaks down Helion's
speedup over each baseline (`torch`, `torch_compile`, and the vLLM op when
installed in the nightly).

The kernels ported from vLLM (`vllm/kernels/helion/ops`) benchmark each fused
Helion kernel under CUDA graphs against a torch-native (unfused, eager)
reference; `silu_mul_fp8` ships an `sm90` heuristic only, the rest ship both
`sm90` and `sm100`.

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

Each kernel module has a `main()` that benchmarks the Helion kernel against
PyTorch eager across the included shape set:

```bash
cd pretuned_kernels/softmax
python softmax.py
```

## Adding a heuristic for new hardware

These kernels ship pretuned heuristics for specific GPU architectures.  To add
another GPU, run the AOT autotune workflow on that hardware against the kernel;
the runner emits a new
`_helion_aot_<kernel>_<device>_<cc>.py` next to the kernel source, which
you commit alongside the existing one(s).  Helion picks the right one at
runtime based on the running GPU's compute capability (with fallback to
older compatible capabilities, e.g. `sm120` → `sm100`).

See the [Ahead-of-Time (AOT) Heuristic Tuning](../docs/deployment_autotuning.md#ahead-of-time-aot-heuristic-tuning)
section of `docs/deployment_autotuning.md` for the end-to-end workflow,
runner CLI, generated artifacts, and runtime fallback rules — including
a worked "Pretuning a kernel for new hardware" walkthrough.
