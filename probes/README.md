# Megakernel probes

This directory contains the self-contained Qwen3, Gemma 4, Nemotron-3, and
DeepSeek-V3 experiments used to evaluate Helion's cross-loop scheduler. Run
commands from the repository root with exactly one idle GPU visible:

```bash
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=0
```

Set `MEGAKERNEL_IDLE_MEMORY_LIMIT_MB` if the driver's idle allocation exceeds
the default 256 MiB. Every comparison below validates outputs, captures CUDA
graphs, and interleaves the megakernel and separate Helion launches in the same
process.

## Recommended comparisons

Qwen3-8B layer:

```bash
python -m probes.qwen3.helion_qwen3_granular_tile_dependency \
  --probe-config \
  --repeats 30 --batch-replays 20
```

The optional `--task-aligned-attention` mode is a dependency-analysis stress
probe; it is not the fastest batch-one lowering.

Gemma 4 E4B layer:

```bash
python -m probes.gemma4.helion_gemma4_e4b_megakernel \
  --layer 0 --config-mode fused --cross-loop-workers 576 \
  --benchmark --repeats 30 --batch-replays 20
```

Gemma 4 26B-A4B MoE sub-layer, preserving the separate GeGLU boundary:

```bash
python -m probes.gemma4.helion_gemma4_a4b_moe_megakernel \
  --batch 1 --route-skew 2 \
  --source-mode assignment_hierarchical_topk_unfused_geglu \
  --config-mode matched --workers 444 --worker-multiplier 4 \
  --num-warps 4 --maxnreg 128 --reduce-block 256 \
  --benchmark --repeats 30 --batch-replays 50
```

Use `--print-lowered` on either Gemma megakernel, or `--dump-triton` on the
Qwen3 probe, to print the complete lowered Triton source.

## Separate Helion baselines

The comparison commands above already run these baselines. They can also be
run alone:

```bash
python -m probes.qwen3.helion_qwen3_layer_baseline --benchmark
python -m probes.gemma4.helion_gemma4_e4b_layer --layer 0 --benchmark
python -m probes.gemma4.helion_gemma4_a4b_moe \
  --batch 1 --route-skew 2 --benchmark
```

Model-specific notes and exploratory Triton scheduling probes live in
`probes/gemma4/`; Qwen3 scheduler ablations and wait profiling live in
`probes/qwen3/`.

To compare the compiler's cache-friendly root-major CLC stream with the
dependency-safe W13-prefix/one-W2-batch ordering ablation, run:

```bash
MEGAKERNEL_CLEAR_L2=1 python -m \
  probes.qwen3.triton_qwen3_ffn_task_stream_order --batch 8
```

The probe writes the complete reordered lowered Triton source to
`/tmp/qwen3_ffn_task_stream_reordered_lowered.py` by default. Pass
`--order batch` to test the more aggressive per-batch W13/W2 loop interchange.
Both reorderings are retained as negative probes; neither is compiler policy.

Nemotron-3 Nano FP8 MoE (B200 production boundaries, including the overlapped
shared-expert stream):

```bash
CUDA_VISIBLE_DEVICES=<idle-gpu> \
python -m probes.nemotron3.helion_nemotron3_nano_moe --benchmark
```

The compiler-generated CLC megakernel uses the same boundaries and compares
against that overlapped baseline in one process:

```bash
CUDA_VISIBLE_DEVICES=<idle-gpu> MEGAKERNEL_CLEAR_L2=1 \
python -m probes.nemotron3.helion_nemotron3_nano_moe_megakernel --benchmark
```

It always writes the complete lowered Triton source to
`/tmp/nemotron3_nano_moe_clc_lowered.py` unless `--lowered-output` selects a
different path. `--dense-routed-activation` is a diagnostic source variant
that removes the redundant valid-row mask while preserving the activation
kernel boundary. The final merge has its own ordinary block-size knob,
`--merge-block`, rather than sharing `--pointwise-block` with every pointwise
root.

Representative DeepSeek-V3 BF16 decode MoE, preserving separate router,
top-k, routed/shared W13, SwiGLU, W2, reduction, and join boundaries:

```bash
CUDA_VISIBLE_DEVICES=<idle-gpu> MEGAKERNEL_CLEAR_L2=1 \
python -m probes.deepseek_v3.helion_deepseek_v3_moe_megakernel \
  --benchmark --repeats 30 --batch-replays 10
```

This compares the compiler-generated CLC megakernel against an overlapped
separate-Helion baseline in the same process and writes the complete lowering
to `/tmp/deepseek_v3_moe_clc_lowered.py`. It is a dependency-scheduling probe,
not an exact reproduction of vLLM's fused grouped-routing implementation.

The checked-in B200 configuration contains tuned `M=1` schedules for the two
routed GEMMs and two shared-expert GEMMs. Retune those kernels with:

```bash
CUDA_VISIBLE_DEVICES=<idle-gpu> \
python -m probes.nemotron3.helion_nemotron3_nano_moe \
  --tune routed_gemm1 \
  --tune routed_gemm2_fused_finalize \
  --tune shared_up_scaled_fp8_mm \
  --tune shared_down_scaled_fp8_mm \
  --tune-effort quick --benchmark
```

Use `--describe` to print the modeled kernel graph without initializing CUDA.
